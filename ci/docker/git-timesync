#!/bin/sh

# This script has been lifted from https://github.com/tst2005/git-timesync/
# and all credits for this belong to @tst2005

# shellcheck disable=SC2004,SC3043,SC2155,SC2039

#### About shellcheck
# I disable:
# - SC2004 (style): $/${} is unnecessary on arithmetic variables.
# Because: I prefere `$( $x - $y ))` than `$(( x - y ))`.

# I disable:
# - SC2039: In POSIX sh, 'local' is undefined.
# - SC3043: In POSIX sh, 'local' is undefined.
# Because:
#   local is too usefull, all modern shell support it.
#   If you really want to run git-timesync on a strict POSIX shell,
#   then remove all local prefixes : 's/local //g'.
#
# I disable :
# - SC2155 (warning): Declare and assign separately to avoid masking return values.
# Because:
#   It not relevant : the return code is not used.
#   I prefer `local foo="$(bar)"` than `local foo;foo="$(bar)"`.
####

set -e

####
# Author: TsT worldmaster.fr <tst2005@gmail.com>
#
# Improvements:
# - dry-run ("-n" flag)
# - pass files to check as argument
# - do not time sync modified files
# - do not time sync untracked files
# - performance improvment
# - be able to apply timesync only on files present in the N last commits
####

####
# The original version of this script can be found at: https://gist.github.com/jeffery/1115504
#
# Helper script to update the Last modified timestamp of files in a Git SCM
# Projects working Copy
#
# When you clone a Git repository, it sets the timestamp of all the files to the
# time when you cloned the repository.
#
# This becomes a problem when you want the cloned repository, which is part of a 
# Web application have a proper cacheing mechanism so that it can re-cache files
# (into a webtree) that have been modified since the last cache.
#
# @see http://stackoverflow.com/questions/1964470/whats-the-equivalent-of-use-commit-times-for-git
#
# Author: Jeffery Fernandez <jeffery@fernandez.net.au>
####

showUsage() {
	echo 'Usage: git-timesync [-n] [-q] [-v] [--] [<paths>...]'
	echo 'Usage: git-timesync [-n] [-q] [-v] -<N>'
	echo
	echo '  -h, --help                 '\
		'Print this help message'
	echo '  -n, --dry-run, --dryrun    '\
		'Perform a dry-run to see which files are OK'\
		'and which ones need to be synchronized'
	echo '  -q, --quiet                '\
		'Quiet mode: drop everything that is OK and show only the files'\
		'which timestamp needs to be synchronized'
	echo '  -v, --verbose              '\
		'Verbose mode: show info for each file (opposite of --quiet)'
	echo '  -1                         '\
		'Apply timesync on files present in the last commit'
	echo '  -23                        '\
		'Apply timesync on files present in the 23 last commits'
	echo '  -N                         '\
		'Apply timesync on files present in the N last commits'\
		'(with 1 <= N <= 9999)'
}

# Get the last revision hash of a particular file in the git repository
getFileLastRevision() {
	git rev-list HEAD -n 1 -- "$1"
}

#getFileMTimeByRef() {
#	git show --pretty=format:%at --abbrev-commit "$1" | head -n 1
#}

getFileMTimeByPath() {
	# shellcheck disable=SC2155
	git rev-list --pretty=format:'date %at' --date-order -n 1 HEAD -- "$1" |
	(
		local IFS=" ";
		# shellcheck disable=SC2034
		while read -r key value _misc; do
			[ "$key" != "date" ] || echo "$value";
		done
	)
}

# Extract the actual last modified timestamp of the file and Update the timestamp
updateFileTimeStamp() {
	# shellcheck disable=SC2155

	# if target does not exists and it's is not a [dead]link, raise an error
	if [ ! -e "$1" ] && [ ! -h "$1" ]; then
		if [ -n "$(git ls-files -t -d -- "$1")" ]; then
			if $verbose; then echo "?  $1 (deleted)"; fi
			return
		fi
		echo >&2 "ERROR: Unknown bug ?! No such target $1"
		return 1
	fi

	local tracked="$(git ls-files -t -c -- "$1")"
	if [ -z "$tracked" ]; then
		if $verbose; then echo "?  $1"; fi
		return
	fi

	# Extract the last modified timestamp
	# Get the File last modified time
	local FILE_MODIFIED_TIME="$(getFileMTimeByPath "$1")"
	if [ -z "$FILE_MODIFIED_TIME" ]; then
		echo "?! $1 (not found in git)"
		return
	fi

	# Check if the file is modified
	local uncommited="$(git ls-files -t -dm -- "$1")"

	# for displaying the date in readable format
	#local FORMATTED_TIMESTAMP="$(date --date="${FILE_MODIFIED_TIME}" +'%d-%m-%Y %H:%M:%S %z')"
	#local FORMATTED_TIMESTAMP="@${FILE_MODIFIED_TIME}"

	# Modify the last modified timestamp
	#echo "[$(date -d "$FORMATTED_TIMESTAMP")]: $1"
	#echo "$FILE_MODIFIED_TIME $1"
	local current_mtime="$(getmtime "$1")"
	if $debug; then
		echo >&2 "DEBUG: $1 (git_time=$FILE_MODIFIED_TIME current_time=$current_mtime delta=$(( ${current_mtime:-0} - ${FILE_MODIFIED_TIME:-0} )))"
	fi
	if [ "$current_mtime" = "$FILE_MODIFIED_TIME" ]; then
		if ${verbose:-true}; then echo "ok $1"; fi
		return
	fi
	if [ -n "$uncommited" ]; then
		echo "C  $1 (modified, not commited, $(( $current_mtime - $FILE_MODIFIED_TIME ))s recent)"
		return
	fi
	if ${dryrun:-true}; then
		echo "!! $1 (desync: $(( $current_mtime - $FILE_MODIFIED_TIME ))s, no change)"
		return
	fi
	echo "!! $1 (desync: $(( $current_mtime - $FILE_MODIFIED_TIME ))s, syncing...)"
	#[ -h "$1" ] && touch -c -h -d "$FORMATTED_TIMESTAMP" -- "$1" || \
	#touch -c -d "$FORMATTED_TIMESTAMP" -- "$1"
	unixtime_touch -c -h -- "$1"
}



# Make sure we are not running this on a bare Repository
is_not_base_repo() {
	case "$(git config core.bare)" in
		false)	;;
		true)
			echo "$(pwd): Cannot run this script on a bare Repository"
			return 1
		;;
		*)	echo "$(pwd): Error appended during core.bare detection. Are you really inside a repository ?"
			return 1
	esac
	return 0
}

updateFileTimeStampInCwd() {
	is_not_base_repo || return

	git ls-files -z \
	| tr '\0' '\n' \
	| (
	while read -r file; do
		if [ -z "$(git ls-files -t -d -- "$file")" ]; then
			updateFileTimeStamp "${file}"
		fi
	done
	)
}

timesyncThisFile() {
	if [ -d "$1" ] && [ ! -h "$1" ]; then # is a real directory (not a symlink to a directory)
		echo "now inside $1"
		# shellcheck disable=SC2015
		( cd -- "$1" && updateFileTimeStampInCwd || true; )
	else
		if $need_check_bare; then
			is_not_base_repo || return 1
			need_check_bare=false
		fi
		updateFileTimeStamp "$1"
	fi
}

# ... for Linux ... and MINGW64 (used by Windows GIT Bash)
linux_unixtime_touch() {
	# shellcheck disable=SC2155
	local FORMATTED_TIMESTAMP="@${FILE_MODIFIED_TIME}"
	touch -d "$FORMATTED_TIMESTAMP" "$@"
}
linux_getmtime() {
	stat -c %Y -- "$1"
}

# ... for FreeBSD and Mac OS X
bsd_unixtime_touch() {
	# shellcheck disable=SC2155
	local FORMATTED_TIMESTAMP="$(date -j -r "${FILE_MODIFIED_TIME}" +'%Y%m%d%H%M.%S')"
	touch -t "$FORMATTED_TIMESTAMP" "$@"
}
bsd_getmtime() {
	stat -f %m -- "$1"
}

################################################################################
############################## MAIN SCRIPT LOGIC ###############################
################################################################################

dryrun=false
verbose=true
debug=false
fromrecent=''
while [ $# -gt 0 ]; do
	case "$1" in
		--) shift; break ;;
		-h|--help) showUsage; exit 0;;
		-n|--dryrun|--dry-run) dryrun=true ;;
		-v) verbose=true ;;
		-q) verbose=false ;;
		-[1-9]|-[1-9][0-9]|-[1-9][0-9][0-9]|-[1-9][0-9][0-9][0-9]) fromrecent="$1" ;;
		--debug) debug=true ;;
		-*) echo >&2 "$0: invalid option $1"; exit 1;;
		*) break
	esac
	shift
done

# Obtain the Operating System
case "${GIT_TIMESYNC_FORCE_UNAME:-$(uname)}" in
	('Linux'|'MINGW64'*)
		unixtime_touch() { linux_unixtime_touch "$@"; }
		getmtime() { linux_getmtime "$@"; }
	;;
	('Darwin'|'FreeBSD')
		unixtime_touch() { bsd_unixtime_touch "$@"; }
		getmtime() { bsd_getmtime "$@"; }
	;;
	(*)
		echo >&2 "Unknown Operating System to perform timestamp update"
		exit 1
	;;
esac

if [ $# -eq 0 ] && [ -z "$fromrecent" ]; then
	# Loop through and fix timestamps on all files in our checked-out repository
	updateFileTimeStampInCwd
else
	need_check_bare=true

	# Loop through and fix timestamps on all specified files
	if [ -n "$fromrecent" ]; then
		git log --format='format:' --name-only "$fromrecent" |
		sort -u |
		while read -r file; do
			[ -n "$file" ] || continue
			[ -e "$file" ] || continue
			timesyncThisFile "$file"
		done
	else
		for file in "$@"; do
			timesyncThisFile "$file"
		done
	fi
fi

