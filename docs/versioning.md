# Experiment versioning

User requirements: use separate Git branches for meaningful iterations; any remote repository must remain private for now. Do not add OpenAI as a coauthor or add AI attribution to commits.

The initial baseline is commit `0040d25` on `main`. Earlier experiments predate Git initialization; this commit records their current implementation, not a reconstructed history.

The active iteration is `experiment/001-cu-ti-temperature-transfer`. Use subsequent numbered experiment branches for changes to architecture, electronic/spin treatment, or experimental protocol. Commit reproducible code, configurations, source manifests, and concise results at milestones. Record the code commit and dataset/checkpoint hashes with completed results. Preserve unsuccessful experiments as well as successful ones.

Raw datasets, virtual environments, training runs, and checkpoints stay outside Git. Review selected reports before adding them; avoid committing generated trajectories or bulky artifacts by default.

The intended remote is `https://github.com/ryanlai666/Semi-E3-MLIP.git`, authorized by the owner and verified private through the GitHub API on 2026-09-21. Verify its actual private visibility before pushing. A local branch name or Git setting does not make a hosted repository private. If creating a repository, explicitly create it private and verify that status before pushing. Do not push to a public destination or change visibility without the user's instruction.
