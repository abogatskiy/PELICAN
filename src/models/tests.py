import torch
import numpy as np
import numpy.matlib
from torch.utils.data import DataLoader
from math import sqrt, pi, cosh
from datetime import datetime

import logging

from ..dataloaders import collate_fn

def permutation_test(model, data):
	logging.info('Beginning permutation test!')

	mask = data['particle_mask']

	# Generate a list of indices for each molecule.
	# We will generate a permutation only for the particles that exist (are not masked.)
	batch_size, particle_size = mask.shape
	perm = 1*torch.arange(particle_size).expand(batch_size, -1)
	for idx in range(batch_size):
		num_particles = (mask[idx, :].long()).sum()
		perm[idx, :num_particles] = torch.randperm(num_particles)

	apply_perm = lambda dim, mat: torch.stack([torch.index_select(mat[idx], dim, p) for (idx, p) in enumerate(perm)])

	assert((mask == apply_perm(0, mask)).all())

	data_noperm = data
	data_perm = {key: apply_perm(0, val) if key in ['Pmu', 'scalars'] else val for key, val in data.items()}
	data_perm['scalars'] = apply_perm(1, data_perm['scalars'])

	outputs_noperm = model(data_noperm)
	outputs_perm = model(data_perm)

	invariance_test = (outputs_perm - outputs_noperm).abs().max()/outputs_noperm.abs().max()

	logging.info('Permutation Invariance test error: {}'.format(invariance_test))
	

def batch_test(model, data):
	logging.info('Beginning batch invariance test!')
	data_split = {key: val.unbind(dim=0) if (torch.is_tensor(val) and val.numel() > 1) else val for key, val in data.items()}
	data_split = [{key: val[idx].unsqueeze(0) if type(val) is tuple else val for key, val in data_split.items()} for idx in range(len(data['is_signal']))]

	outputs_split = torch.cat([model(data_sub) for data_sub in data_split])
	outputs_full = model(data)
	invariance_test = (outputs_split - outputs_full).abs().max()/outputs_full.abs().mean()

	logging.info('Batch invariance test error: {}'.format(invariance_test))


def ir_data(data_irc, num_particles, alpha):
### 
# Add num_particles random four-momenta to the data array with alpha specifying the scaling relative to O(1)
###
	batch_size = data_irc['Nobj'].shape[0]
	device = data_irc['Pmu'].device
	dtype = data_irc['Pmu'].dtype
	zero = torch.tensor(0.)

	rand_pmus = alpha * torch.add(2*torch.rand(batch_size, num_particles, 4, dtype = data_irc['Pmu'].dtype, device = data_irc['Pmu'].device), -1)
	rand_pmus_sq = torch.pow(rand_pmus, 2)
	data_irc['Pmu'] = torch.cat((data_irc['Pmu'], rand_pmus), 1)
	s = data_irc['Pmu'].shape
	particle_mask = data_irc['Pmu'][...,0] != 0.
	edge_mask = particle_mask.unsqueeze(1) * particle_mask.unsqueeze(2)
	labels = torch.zeros([s[0],s[1],s[1],2])
	if data_irc['scalars'][0,0,0,0] == 1.:      #this assumes that the first two scalars are 1 only for beams
		# labels = torch.cat((torch.ones(s[0], 2), torch.zeros(s[0], s[1])), dim=1)
		# labelsi = labels.unsqueeze(1).expand(s[0], s[1]+num_particles, s[1]+num_particles)
		# labelsj = labels.unsqueeze(2).expand(s[0], s[1]+num_particles, s[1]+num_particles)
		labels[:,[0,1],:,1]=1.
		labels[:,:,[0,1],0]=1.
		labels = torch.where(edge_mask.unsqueeze(-1), labels, zero)
	data_irc['scalars'] = labels.to(dtype=data_irc['Pmu'].dtype)
	data_irc['Nobj'] = data_irc['Nobj'] + torch.ones(batch_size, dtype=torch.int) * num_particles
	data_irc['particle_mask'] = particle_mask
	data_irc['edge_mask'] = edge_mask
	# if data_irc['scalars'].shape[2]==7: breakpoint()
	return data_irc

def c_data(data_irc):
	### Split num_particles leading four-momenta into two equal halves
	indices = [2,3]  #which input particles to split. [2,3] means the two leading particles, skipping the two beams [0,1]
	batch_size = data_irc['Nobj'].shape[0]
	event_size = data_irc['Pmu'].shape[1]
	num_particles = len(indices)
	device = data_irc['Pmu'].device
	dtype = data_irc['Pmu'].dtype
	zero = torch.tensor(0.)

	original_momenta = data_irc['Pmu'][:,indices,:]
	data_irc['Pmu'][:,indices,:]=original_momenta/2.
	data_irc['Pmu']=torch.cat((data_irc['Pmu'], original_momenta/2), dim=1)
	data_irc['scalars'] = torch.cat((data_irc['scalars'], torch.zeros([batch_size,num_particles,event_size,2]).to(dtype=data_irc['Pmu'].dtype)),-3)
	data_irc['scalars'] = torch.cat((data_irc['scalars'], torch.zeros([batch_size,event_size+num_particles,num_particles,2]).to(dtype=data_irc['Pmu'].dtype)),-2)
	data_irc['Nobj'] = data_irc['Nobj'] + torch.ones(batch_size, dtype=torch.int) * num_particles
	data_irc['particle_mask'] = data_irc['Pmu'][...,0] != 0.
	data_irc['edge_mask'] = data_irc['particle_mask'].unsqueeze(1) * data_irc['particle_mask'].unsqueeze(2)

	return data_irc

def irc_test(model, data):
	logging.info('IRC safety test!')

	data_irc = {key: val.clone() if type(val) is torch.Tensor else val for key, val in data.items()}

	outputs = model(data)
	outputs_ir=[]
 
	# First check IR safety (injection of new small momenta). This one is easier to enforce in a model -- 
	# -- simply make sure that the outputs of the network are small whenever the inputs are.

	alpha_range = 10.**np.arange(-6,-2,step=1)
	max_particles=2
	for alpha in alpha_range:
		temp = []
		for num_particles in range(1, max_particles + 1):
			data_ir = ir_data(data_irc, num_particles, alpha)
			temp.append(model(data_ir))
		outputs_ir.append([alpha, torch.stack(temp,0)])

	data_irc = {key: val.clone() if type(val) is torch.Tensor else val for key, val in data.items()}

	ir_test = [(alpha, (output_ir - outputs.unsqueeze(0).repeat(max_particles, 1, 1)).abs().amax((1,2))/outputs.abs().mean()) for (alpha, output_ir) in outputs_ir]
	logging.info('IR safety test deviations (format is (order of magnitude of momenta: [1 particle, 2 particles, ...])):')
	for alpha, output in ir_test:
		logging.warning('{:0.5g}, {}'.format(alpha, output.data.cpu().detach().numpy()))


	# Now Colinear safety (splitting a present four-momentum into two identical or almost identical halves).
	# Can also be viewed as creating 2 particles with identical spatial momenta but half of the original energy.
	# This symmetry cannot be explicitly enforced in a nonlinear network because it requires some sort of linearity in energy.
	# Instead, we simply test for it and hope that training a network improves C-safety.
	data_c = c_data(data_irc)
	outputs_c = model(data_c)
	c_test = (outputs_c - outputs).abs().max()/outputs.abs().mean()
	logging.info('C safety test deviations: {}'.format(c_test.data.cpu().detach().numpy()))

def gpu_test(model, data, t0):
	logging.info('Starting the computational cost test!')
	device = model.device
	t1 = datetime.now()
	logging.info(f'{"Collation time:":<80} {(t1-t0).total_seconds()}s')
	if model.device=='gpu':
		mem_init = torch.cuda.memory_allocated(device)
		logging.info(f'{"Initial GPU memory allocation:":<80} {mem_init}')
		output = model(data)
		t2 = datetime.now()
		mem_fwd =  torch.cuda.memory_allocated(device)
		logging.info(f'{"Forward pass time:":<80} {(t2-t1).total_seconds()}s')
		logging.info(f'{"Memory after forward pass:":<80} {mem_fwd}')
		logging.info(f'{"Memory consumed by forward pass:":<80} {mem_fwd-mem_init}')
		ouput.sum().backward()
		t3 = datetime.now()
		mem_bwd =  torch.cuda.memory_allocated(device)
		logging.info(f'{"Backward pass time:":<80} {(t3-t2).total_seconds()}s')
		logging.info(f'{"Memory after backward pass:":<80} {mem_bwd}')
		logging.info(f'{"Total batch time:":<80} {(t3-t0).total_seconds()}s')
		del output
		with torch.no_grad:
			t4 = datetime.now()
			mem_init = torch.cuda.memory_allocated(device)
			logging.info(f'{"Initial GPU memory allocation:":<80} {mem_init}')
			output = model.forward(data)
			t5 = datetime.now()
			mem_fwd =  torch.cuda.memory_allocated(device)
			logging.info(f'{"Inference time:":<80} {(t5-t4).total_seconds()}s')
			logging.info(f'{"Inference memory usage (over initial):":<80} {mem_fwd-mem_init}')
	else:
		output = model(data)
		t2 = datetime.now()
		logging.info(f'{"Forward pass time:":<80} {(t2-t1).total_seconds()}s')
		output.sum().backward()
		t3 = datetime.now()
		logging.info(f'{"Backward pass time:":<80} {(t3-t2).total_seconds()}s')
		logging.info(f'{"Total batch time:":<80} {(t3-t0).total_seconds()}s')
		with torch.no_grad():
			t4 = datetime.now()
			output = model.forward(data)
			t5 = datetime.now()
			logging.info(f'{"Inference time:":<80} {(t5-t4).total_seconds()}s')




def tests(model, dataloader, args, tests=['permutation','batch','irc']):
	if not args.test:
		logging.info("WARNING: network tests disabled!")
		return

	logging.info("Testing network for symmetries:")
	model.eval()
	
	t0 = datetime.now()
	data = next(iter(dataloader))

	if 'gpu' in tests:
		gpu_test(model, data, t0)
	if 'permutation' in tests:
		permutation_test(model, data)
	if 'batch' in tests:
		batch_test(model, data)
	if 'irc' in tests:
		irc_test(model, data)

	logging.info('Test complete!')