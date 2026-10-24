# Prompts Log

> **Purpose:** Saved prompt templates and agent instructions that were useful — so we can re-run them or give them to other agents/workspaces.

## Historical records (dated)
- `2026-08-06.md` — Wave 4 prompts (P6-P8: mixed-backend debug, post-apt NVIDIA maintenance, cluster health check)

---

## P1: PR Merge Campaign Master Prompt (2026-08-05)

> "Here is what I would like to do: I would like to go one by one top down and start merging all of those changes into this repository, as well as running major major loops on each code, using all the skills we have available to verify that the code that each of those pull requests as we one by one merge in are really good and well written, and if we find (you set a threshold) that the code that is in that pull request maybe needs me to interact with you as an agent to make some changes or decisions then we will move on to the next one. By merge I simply mean you will look at the code changes in that pull request and you will make those code changes locally and make them commits, and the pull request notes, comments and numbers and information of who made it will be added to the title and comment section of the commit. You will also include the urls. You are not pushing anything, you are just staying on the current branch and making the commits on this branch."

**Result:** 53 PRs ported, 3 waves, all with attribution.

## P2: Track-A/Track-B Research Prompt (2026-08-05)

> "Make a copy of the current branch and then you will make a branch specifically for the changes that will be pulled in and verify every single line of code for this set of PRs making sure that they will actually be useful and helpful with our current hardware setup..."

**Result:** track-a/track-b branches created (later abandoned when user pivoted).

## P3: Documentation-Expansion Prompt (2026-08-06)

> "Don't forget about all the ways we can use the plansANDprompts folder and all the files and stuff in there. You're supposed to put things in there as well. So do that and fill in details of everything we have been working on, questions asked, and I literally mean everything..."

**Result:** Full folder structure expansion (decisions/, learnings/, tasks/, references/, branches/, notes/, examples/, prompts/).

## P4: Commit Attribution Standard (2026-08-06)

Every commit must include: WHAT changed / WHY (user request vs PR port vs bug fix) / source attribution (PR #, author, URL) / if fixing someone else's code, reference their PR/commit/link / verification notes.

## P5: Cross-workspace Handoff Prompt Template

When giving another agent context about this project, include:
1. Branch name + commit count + latest commit
2. Cluster hardware (machines, backends)
3. Known gaps (Linux box pynvml/nvidia-smi)
4. Eject behavior (no auto-delete)
5. Point to promptsANDplans README + session-tracking
