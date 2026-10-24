#! /bin/bash

#SBATCH -p genx
#SBATCH -c 1
#SBATCH --mem 2GB
#SBATCH --time=168:00:00
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err

set -e
set -u

module load modules/2.0-20220630
module load postgresql/12.2

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8


# Port and password to connect
# We don't use default port to avoid conflicts with other users on system
PORT=35719
PASSWORD=asoetuh

# Setup database directory
PG_DATA_DIR=/tmp/pgsql/data
rm -rf $PG_DATA_DIR
mkdir -p $PG_DATA_DIR
initdb -D $PG_DATA_DIR --auth-host=scram-sha-256 --auth-local=trust --pwfile=<(echo $PASSWORD)

# Set-up host files to be accessible from everywhere
echo "listen_addresses='*'" >> $PG_DATA_DIR/postgresql.conf
echo "host all all 0.0.0.0/0 scram-sha-256" >> $PG_DATA_DIR/pg_hba.conf

pg_ctl start -D $PG_DATA_DIR -o "-p $PORT"

# By default, create db from username
createdb --port $PORT --no-password

# Install trap for sigterm
_term() {
    echo "Dumping last state"
    pg_dump --port $PORT $USER | gzip > backup.gz
    echo "Stopping server"
    pg_ctl stop -D $PG_DATA_DIR
    echo "Cleaning up temporary directory"
    rm -rf "$PG_DATA_DIR"
}

trap _term EXIT

# Keep rolling backup of DB state just in case.
# No strictly necessary as we also save on exit
echo "Saving db state every minute"
while true
do
    # Dump the default db to backup.gz in case we want to restore at later time.
    pg_dump --port $PORT $USER | gzip > backup.gz
    sleep 60
done