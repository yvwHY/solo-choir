# Skill provenance & licenses

Source material for the skills installed under `.claude/skills/`. Kept here for
attribution, license compliance, and thesis provenance.

## Verbatim originals in this folder
- `superpowers-brainstorming.SKILL.md` — upstream source for our adapted `/brainstorm`.
- `superpowers-writing-plans.SKILL.md` — upstream source for our adapted `/write-plan`.

Our shipped versions in `.claude/skills/brainstorming/` and `.claude/skills/writing-plans/`
are **adapted**: superpowers-internal sub-skill references removed, the TDD
verification cycle generalized to this project's verify-by-ear / byte-identical-diff /
RTF reality, and save paths pointed at `docs/`. Diff these files against
the shipped ones to see exactly what changed.

## Sources & licenses

| Skill (shipped) | Upstream | License | Verbatim or adapted |
|---|---|---|---|
| `grill-me`, `grilling` | github.com/mattpocock/skills | MIT | verbatim |
| `ponytail`, `ponytail-review` | github.com/DietrichGebert/ponytail | MIT | verbatim (license header kept in each SKILL.md) |
| `brainstorming` (`/brainstorm`) | github.com/obra/superpowers | MIT | adapted (original saved here) |
| `writing-plans` (`/write-plan`) | github.com/obra/superpowers | MIT | adapted (original saved here) |
| CLAUDE.md "Working style" section | github.com/multica-ai/andrej-karpathy-skills | no explicit license | paraphrased, not copied |

All four code-bearing sources are MIT-licensed; attribution retained here and in
the shipped SKILL.md headers where present. The Karpathy guidelines repo carries
no explicit license, so its principles were paraphrased into CLAUDE.md rather than
copied verbatim.
