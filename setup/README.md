# Setting Up Git Collaboration

[SSH Keys](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)
Note:
1. Empty your password for SSH keys
2. Don't forget to add it to [your account](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/)

## Set your identity

  ```bash
  $ git config --global user.email "<your_github_email>"
  $ git config --global user.name "<your_github_username>"
  ```

## Set Upstream

  ```bash
  $ git remote add upstream git@github.com:sanadhis/IST_ML_Project1.git
  ```

## Updating your local project with latest version of upstream
  
  ```bash
  $ git fetch upstream
  $ git rebase upstream/master
  ```