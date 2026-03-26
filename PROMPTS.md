# PROMPTS

This file records the main documentation-oriented prompts used during the current refinement pass of the `SuperDog` repository. It is not a full historical build log of the entire project; it is a curated log for README / RUNME / artifact presentation work.

## Prompt Log

1. Create TensorBoard-based training plots for the main SAC navigation metrics and save them as documentation-ready images.
2. Improve the visual quality of the plots and render them with standard plotting tools so they are suitable for presentation.
3. Rewrite `RUNME.md` into a cleaner install-and-run guide and separate it from the broader project overview.
4. Expand `README.md` into a stronger project document that explains the SAC navigation setup, curriculum learning, observations, rewards, and experiments.
5. Move the experiments into the main README and keep `RUNME.md` focused only on installation and execution commands.
6. Replace the inline HTML architecture block with a single clean architecture image.
7. Add a mathematical SAC section with the main objective, critic target, actor update, change-of-variables correction, entropy tuning, and target-network update formulas.
8. Convert actor and critic observation descriptions from HTML tables into standard Markdown tables.
9. Compare the README against the provided good example and add the missing presentation sections, including dependencies, commands, prompt log, references, and a sim-to-real placeholder.
10. Clarify what the axes mean in each experiment figure so the plots can be read without external context.

## Notes

- The operational command reference lives in [RUNME.md](RUNME.md).
- The main project overview and experiment discussion live in [README.md](README.md).
- The documentation figures used by the README are stored under `docs/experiments/`.
