# Git & GitHub

## Table of Contents

1. [Core Concepts](#1-core-concepts)
   - 1.1 [What is Git?](#11-what-is-git)
   - 1.2 [Key Terms](#12-key-terms)
2. [Basic Workflow](#2-basic-workflow)
   - 2.1 [Uploading Code to GitHub](#21-uploading-code-to-github)
   - 2.2 [Essential Commands](#22-essential-commands)
3. [Branching](#3-branching)
   - 3.1 [What is a Branch?](#31-what-is-a-branch)
   - 3.2 [Branch Commands](#32-branch-commands)
   - 3.3 [Merge vs Rebase](#33-merge-vs-rebase)
4. [Team Collaboration](#4-team-collaboration)
   - 4.1 [Fork & Pull Request](#41-fork--pull-request)
5. [Additional Commands](#5-additional-commands)
   - 5.1 [Modifying Commits](#51-modifying-commits)
   - 5.2 [Undoing Changes](#52-undoing-changes)
   - 5.3 [Saving Work Temporarily](#53-saving-work-temporarily)

---

## 1. Core Concepts

### 1.1 What is Git?

Git is a **distributed version control system** that tracks changes to files over time. Every developer has a full copy of the repository (local), and changes are synchronized with a shared remote repository (e.g., GitHub).

```
Local machine                        Remote (GitHub)
┌──────────────────────┐             ┌──────────────┐
│  Working Directory   │             │              │
│  (edited files)      │             │  Repository  │
│         ↓ git add    │             │              │
│  Staging Area        │             │              │
│  (files to commit)   │             │              │
│         ↓ git commit │             │              │
│  Local Repository    │──git push──▶│              │
│  (.git folder)       │◀──git pull──│              │
└──────────────────────┘             └──────────────┘
```

### 1.2 Key Terms

| Term               | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| **Repository**     | A project folder tracked by Git, containing all history             |
| **Commit**         | A snapshot of the project at a specific point in time               |
| **Branch**         | An independent line of development                                   |
| **HEAD**           | A pointer to the current local branch you are working on            |
| **Remote**         | A reference to a repository hosted on a server (e.g., GitHub)       |
| **Staging Area**   | A buffer between working directory and commit; files added here will be included in the next commit |

---

## 2. Basic Workflow

### 2.1 Uploading Code to GitHub

Step-by-step process to push a local project to GitHub:

```
1. Initialize git in the project folder     → git init
2. Select files to include in the commit    → git add <file>
3. Bundle the selected files with a message → git commit -m "description"
4. Create a repository on github.com        → (done on the website)
5. Link the local folder to the remote repo → git remote add <name> <url>
6. Upload commits to the remote repository  → git push <name> <branch>
```

### 2.2 Essential Commands

---

#### `git init`
Initialize a Git repository in the current folder.

```bash
git init
```

- Creates a hidden `.git` folder that stores version history, remote URLs, and configuration.
- Only one `.git` folder should exist per project root.
- When you `clone` a remote repository, a local `.git` is created automatically.

---

#### `git add`
Stage files to be included in the next commit.

```bash
git add <filename>      # stage a specific file
git add .               # stage all changed files in the current directory
git add -p              # interactively stage chunks of changes
```

---

#### `git commit`
Save a snapshot of the staged files with a descriptive message.

```bash
git commit -m "message"
```

- A commit records **what** changed and **when**.
- Write messages in the imperative mood: `"Fix login bug"`, not `"Fixed login bug"`.

---

#### `git log`
View the commit history.

```bash
git log                  # full history
git log --oneline        # compact, one line per commit
git log --oneline --graph --all  # visual branch graph
```

---

#### `git remote add`
Link the local repository to a remote repository on GitHub.

```bash
git remote add <nickname> <url>

# Example
git remote add origin https://github.com/username/repo.git
```

- `origin` is the conventional nickname for your primary remote.
- View existing remotes: `git remote -v`

---

#### `git push`
Upload local commits to the remote repository.

```bash
git push <nickname> <branch>

# Example
git push origin main
```

- Use `-u` on first push to set the upstream: `git push -u origin main`
- After that, `git push` alone is sufficient.

---

#### `git clone`
Download a remote repository to your local machine.

```bash
git clone <url>          # clone into a new folder named after the repo
git clone <url> .        # clone into the current directory
```

---

#### `git pull`
Fetch updates from the remote and merge them into the current branch. Equivalent to `git fetch` + `git merge`.

```bash
git pull <nickname> <branch>

# Example
git pull origin main
```

---

## 3. Branching

### 3.1 What is a Branch?

A branch is an **independent line of development**. When two or more developers work on features that may conflict, each works on their own branch and merges later.

```
main   ──●──●──────────────────●── (merge)
              \               /
feature        ●──●──●──●──●
```

- **HEAD** always points to the branch you are currently on.
- The default branch is typically named `main` (or `master` in older repos).

### 3.2 Branch Commands

---

#### `git branch`
Manage branches.

```bash
git branch                  # list all local branches
git branch <name>           # create a new branch
git branch -d <name>        # delete a branch (safe — only if merged)
git branch -D <name>        # force delete a branch
git branch -a               # list all local and remote branches
```

---

#### `git checkout` / `git switch`
Move HEAD to a different branch.

```bash
git checkout <branch>       # switch to an existing branch
git checkout -b <branch>    # create and switch in one step

# Modern alternative (Git 2.23+)
git switch <branch>
git switch -c <branch>      # create and switch
```

---

#### `git merge`
Integrate another branch into the current branch (union of both branches).

```bash
# Switch to the branch you want to merge INTO, then:
git merge <branch>

# Example: merge feature into main
git checkout main
git merge feature
```

If the same lines were edited in both branches, a **merge conflict** occurs. Git marks the conflicting sections and you must resolve them manually, then `git add` and `git commit`.

```
<<<<<<< HEAD
your changes
=======
incoming changes
>>>>>>> feature
```

### 3.3 Merge vs Rebase

| Feature         | `merge`                                   | `rebase`                                  |
|-----------------|-------------------------------------------|-------------------------------------------|
| History shape   | Preserves all branch history (non-linear) | Rewrites history into a single linear line |
| Commit graph    | Creates a merge commit                    | No merge commit                            |
| When to use     | Integrating finished features             | Keeping feature branch up to date with main |
| Safety          | Safe on shared branches                   | Never rebase commits already pushed to a shared branch |

```bash
# Rebase feature branch onto main (replay feature commits on top of main)
git checkout feature
git rebase main
```

```
Before rebase:                After rebase:
main   ──●──●──●              main   ──●──●──●
              \                                \
feature        ●──●                  feature    ●──●
```

---

## 4. Team Collaboration

### 4.1 Fork & Pull Request

Used when contributing to a repository you don't have write access to (open source, cross-team collaboration). Done on the GitHub website.

**Workflow:**

```
1. Fork — copy the target repository to your own GitHub account
2. Clone — download your fork to your local machine
3. Branch — create a feature branch
4. Commit & Push — make changes and push to your fork
5. Pull Request — request to merge your branch into the original repository
6. Review & Merge — the repo owner reviews and merges (or requests changes)
```

```
Original repo (upstream)          Your fork (origin)
┌──────────────────────┐   fork   ┌──────────────────────┐
│  owner/project       │────────▶ │  you/project         │
│                      │          │                      │
│                      │◀─────────│  (pull request)      │
└──────────────────────┘  merge   └──────────────────────┘
                                           ▲ push
                                  ┌──────────────────────┐
                                  │  Local machine       │
                                  └──────────────────────┘
```

Keep your fork in sync with the upstream:

```bash
git remote add upstream <original-repo-url>
git fetch upstream
git merge upstream/main
```

---

## 5. Additional Commands

### 5.1 Modifying Commits

#### `git commit --amend`
Add forgotten changes to the most recent commit, or edit its message.

```bash
git add <forgotten-file>
git commit --amend -m "updated message"
```

> Only amend commits that have **not** been pushed to a shared branch — rewriting published history causes problems for other developers.

---

#### `git cherry-pick`
Apply a specific commit from another branch onto the current branch.

```bash
git cherry-pick <commit-hash>
```

Useful when you want one specific fix from another branch without merging the entire branch.

---

### 5.2 Undoing Changes

#### `git reset`
Move HEAD (and optionally the staging area / working directory) back to an earlier commit.

```bash
git reset --soft  <commit>   # move HEAD only; staged changes preserved
git reset --mixed <commit>   # move HEAD + unstage changes (default)
git reset --hard  <commit>   # move HEAD + discard all changes (destructive)
```

> `--hard` permanently discards changes. Use with caution.

---

#### `git revert`
Create a **new commit** that undoes the changes of a specified commit. Safe for shared branches because it does not rewrite history.

```bash
git revert <commit-hash>
```

| Command      | Rewrites history | Safe on shared branch |
|--------------|:----------------:|:---------------------:|
| `git reset`  | Yes              | No                    |
| `git revert` | No               | Yes                   |

---

### 5.3 Saving Work Temporarily

#### `git stash`
Temporarily shelve uncommitted changes so you can switch context without committing incomplete work.

```bash
git stash                    # stash current changes
git stash list               # view all stashed entries
git stash pop                # restore the most recent stash and remove it
git stash apply stash@{1}   # restore a specific stash without removing it
git stash drop stash@{0}    # delete a specific stash
git stash clear              # delete all stashes
```

**Common use case:**

```bash
# You're mid-feature when an urgent bug is reported on main
git stash                    # save current work
git checkout main            # switch to main
# fix the bug, commit
git checkout feature         # return to feature branch
git stash pop                # restore your work
```
