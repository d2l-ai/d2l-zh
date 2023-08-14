function setup_git {
    # Turn off logging
    set +x
    mkdir -p $HOME/.ssh
    echo "yes" | ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

    # Retrieve the SSH key securely from AWS Secrets Manager
    GIT_SSH_KEY=$(aws secretsmanager get-secret-value --secret-id d2l_bot_github --query SecretString --output text --region us-west-2)

    # Write the SSH key to a file
    echo "$GIT_SSH_KEY" > $HOME/.ssh/id_rsa
    chmod 600 $HOME/.ssh/id_rsa

    git config --global user.name "d2l-bot"
    git config --global user.email "100248899+d2l-bot@users.noreply.github.com"

    echo "Successfully Configured Bot"
}
