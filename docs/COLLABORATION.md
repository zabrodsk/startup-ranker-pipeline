# Collaboration Workflow

This project is set up to be shared through GitHub. Each collaborator should work from their own local clone and use Codex against that local folder.

## One-time setup

1. Add each collaborator to the GitHub repository.
2. Clone the repository on each machine:

```bash
git clone https://github.com/zabrodsk/Rockaway-Deal-Intelligence.git
cd Rockaway-Deal-Intelligence
```

3. Create a local environment file:

```bash
cp .env.example .env
```

4. Fill in secrets locally. Do not commit `.env`.

## Working at the same time

Never share a live working directory through iCloud, Dropbox, or similar sync tools while both people are editing. Use GitHub to exchange changes.

Recommended flow for every task:

```bash
git checkout main
git pull origin main
git checkout -b feature/short-description
```

After making changes:

```bash
git add .
git commit -m "Short description"
git push -u origin feature/short-description
```

Then open a pull request, review it, merge it, and update local `main` before starting the next task.

## Suggested conventions

- Keep branches focused on one task.
- Pull `main` before starting new work.
- Avoid long-lived branches.
- Commit early if you need to preserve work before switching tasks.
- Update `.env.example` when required configuration changes.
- Never commit `.env`, credentials, or generated local junk files.

## ChatGPT / Codex

If you also want shared AI context:

- Share the ChatGPT Project with your collaborator.
- Keep in mind that project chats are not a replacement for git history.
- Use the shared project for instructions, notes, and files.
- Use GitHub for the actual source-of-truth codebase.
