# Experiment versioning

User requirements: use separate Git branches for meaningful iterations. Keep the remote private while the current work is in progress. On 2026-09-21 the owner explicitly authorized changing it to public after the remaining training, benchmarks, visual checks, and final documentation are complete. Do not add OpenAI as a coauthor or add AI attribution to commits.

The initial baseline is commit `0040d25` on `main`. Earlier experiments predate Git initialization; this commit records their current implementation, not a reconstructed history.

The active iteration is `experiment/001-cu-ti-temperature-transfer`. Use subsequent numbered experiment branches for changes to architecture, electronic/spin treatment, or experimental protocol. Commit reproducible code, configurations, source manifests, and concise results at milestones. Record the code commit and dataset/checkpoint hashes with completed results. Preserve unsuccessful experiments as well as successful ones.

Raw datasets, virtual environments, training runs, and checkpoints stay outside Git. Review selected reports before adding them; avoid committing generated trajectories or bulky artifacts by default.

The intended remote is `https://github.com/ryanlai666/Semi-E3-MLIP.git`, authorized by the owner and verified private through the GitHub API on 2026-09-21. Verify its actual private visibility before pushing. A local branch name or Git setting does not make a hosted repository private. If creating a repository, explicitly create it private and verify that status before pushing. The owner's later instruction authorizes one final private-to-public change after completion. Verify the actual visibility after changing it; do not publish early.
