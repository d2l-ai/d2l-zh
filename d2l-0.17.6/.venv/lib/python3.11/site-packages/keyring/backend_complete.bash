# Complete keyring backends for `keyring -b` from `keyring --list-backends`
# # keyring -b <TAB>
# keyring.backends.chainer.ChainerBackend keyring.backends.fail.Keyring ...

_keyring_backends() {
	local choices
	choices=$(
		"${COMP_WORDS[0]}" --list-backends 2>/dev/null |
			while IFS=$' \t' read -r backend rest; do
				printf "%s\n" "$backend"
			done
	)
	compgen -W "${choices[*]}" -- "$1"
}
