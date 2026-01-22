# Complete keyring backends for `keyring -b` from `keyring --list-backends`
# % keyring -b <TAB>
# keyring priority
# keyring.backends.chainer.ChainerBackend   10
# keyring.backends.fail.Keyring             0
# ...                                       ...

backend_complete() {
	local line
	while read -r line; do
		choices+=(${${line/ \(priority: /\\\\:}/)/})
	done <<< "$($words[1] --list-backends)"
	_arguments "*:keyring priority:(($choices))"
}
