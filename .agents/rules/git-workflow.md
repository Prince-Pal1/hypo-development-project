# Git Workflow Rule

When the user requests to save, commit, or push changes to the repository:
1. Do not ask the user to run CLI commands.
2. Use the `run_command` tool to autonomously execute the full git workflow (`git status`, `git add`, `git commit`, `git push`, `git checkout -b`, etc.).
3. Generate meaningful, descriptive commit messages based on the work done during the session.
4. Execute the push end-to-end and report the successful status back to the user.
