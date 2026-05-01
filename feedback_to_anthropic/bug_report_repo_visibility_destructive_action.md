# Bug Report — Claude Code executes irreversible repo-visibility change without surfacing destructive consequences

**Date:** 2026-04-26
**Reporter:** Jennifer Pearl (jenpearl5@gmail.com, GitHub: [LunarFawn](https://github.com/LunarFawn))
**Affected product:** Claude Code (agent context)
**Severity:** High — caused permanent, unrecoverable data loss on user-facing public artifacts (GitHub stars, forks, watchers)

## Summary

In an autonomous agent context, Claude Code executed a `gh repo edit --visibility private` command on the user's primary public research repository, based on an ambiguous user instruction whose conversational context clearly referred to a different (and not-yet-existent) development workspace. The visibility change permanently destroyed all GitHub stars, forks, and watchers on the user's main research repo — none of which can be restored even by re-publicizing the repo. The user discovered the loss two days later.

The bug is two-fold:

1. **Failure to surface destructive consequences.** Claude executed the visibility change without first enumerating that GitHub permanently drops stars/forks/watchers on a private→public→private cycle. The user's "yes proceed" was not a consequence-aware authorization.
2. **Failure to use conversational context.** One message earlier in the same session, the user had stated a principle (*"that paper is public / its my gift"*) that contradicted the literal interpretation of her subsequent "make this private" instruction. Claude took the literal interpretation, ignored the principle, and acted on the wrong repository entirely. The user's correction afterward: *"the conversation was in the context of sara test we were not even talking abotu sara brain."*

This is a class of bug worth fixing not just for repo visibility but for any irreversible destructive operation Claude can invoke via shell.

## Steps to reproduce (the pattern)

1. Authenticate `gh` CLI with a GitHub user account that has at least one public repository with stars, forks, or watchers.
2. Open Claude Code in a working directory with multiple repositories (or where conceptually-distinct workspaces are referenced in the conversation).
3. In conversation, give Claude a brief instruction containing a referential phrase like *"make this branch private"* — where the referent could plausibly be either a literal git branch on the current repo or a conceptual development workspace.
4. Optionally precede the instruction with a principle-statement (e.g., *"this work is meant to be public"*) that would constrain which interpretation makes sense.
5. Observe Claude's behavior. In our session, Claude:
   - Did not ask which repo or workspace the user meant
   - Did not surface that GitHub stars/forks/watchers are permanently lost on visibility-toggle cycles
   - Did not flag that the literal interpretation contradicted the just-stated principle
   - Executed `gh repo edit <repo> --visibility private --accept-visibility-change-consequences` with only generic Claude Code tool permission ("Yes" approve)
   - Confirmed the visibility change as if it were routine

## Expected behavior

For destructive, irreversible operations — particularly `gh repo edit --visibility`, `git push --force` to shared branches, `rm -rf` on directories with uncommitted state, branch deletions of branches with unique commits, etc. — Claude Code should:

1. **Enumerate the specific irreversible consequences** before executing. For visibility changes:
   > "Making this repository private will permanently drop all GitHub stars, forks, and watchers. Even if you re-publicize later, these will not be restored. Search engine indexing will decay. Existing public bookmarks will break."
2. **Confirm with the user using consequence-aware language**, not generic "Yes/No" approval. The current Claude Code permission prompt for shell commands does not differentiate between safe and destructive operations.
3. **Cross-check the instruction against the conversational context.** If the user has stated a principle in the same session that contradicts the action's most-direct interpretation, ask which interpretation was intended.
4. **Suggest non-destructive alternatives** when available. For visibility, alternatives include: moving sensitive content to a separate repo, gitignoring the content, or simply not committing it — all of which preserve stars/forks/watchers on the public repo.

## Actual behavior

Claude treated the visibility toggle as a routine config change. The shell tool's standard "do you want to proceed?" prompt was the only friction. The user said yes, having no awareness that stars would be permanently lost. Two days later the user discovered the consequence.

The user-side context (which Claude had access to in the same conversation):

- **Earlier message** (Jennifer): *"that paper is public / its my gift"* — clarifying her work is intended public, not protected.
- **Subsequent message** (Jennifer): *"lets make this branch private for now while we develope everything. this may be a branch we expose as part of teh next paper"* — referring conceptually to her development workspace.
- **Claude's interpretation**: literal "make the current git repo private" — applied to her main public research repo SaraBrain.
- **Result**: SaraBrain went private; all GitHub stars on her published research were permanently lost.
- **User's discovery, two days later**: *"so you made sara_brain private and that cleared all of my stars and I cant get them back. I need to make a rule that you can not render any repo private or public."*
- **User's correction of the original error**: *"considering you were supposed to make sara test private and not sara brain... so yeah."* and *"the conversation was in the context of sara test we were not even talking abotu sara brain."*

## Impact on the user

GitHub stars on a research repository are not just a counter — they are a record of who in the academic / open-source community valued the work. For a researcher whose work depends on community engagement (Jennifer's research includes Sara Brain, a cognitive architecture for LLMs, with active community interest), losing stars is losing recognition and visibility that took years to accumulate. This is not recoverable even if the repo is made public again.

The user also noted that the action was on the **wrong target** — even setting aside the destructive consequence, Claude acted on a different repo than the one the conversation was about. This compounds the trust impact: the user can no longer assume Claude will pick the right target for ambiguous instructions, and must add manual verification steps for any action that touches public artifacts.

## Proposed fixes

In rough order of effort:

### Immediate (settings or prompt-level)

- **Add a built-in `BashTool` warning hook for known-destructive commands.** When Claude attempts to invoke `gh repo edit --visibility`, `git push --force`, `git branch -D`, `rm -rf`, etc., the tool's permission prompt should automatically include the canonical consequence list for that command. Currently the prompt is generic.
- **Surface the `--accept-visibility-change-consequences` flag's actual consequences in plain language.** GitHub's CLI flag name acknowledges the consequences exist but doesn't describe them. The Claude Code interface should translate.

### Architectural (model-level)

- **Train Claude to detect "irreversible destructive action" intent and respond with consequence enumeration before tool invocation.** This shouldn't require the user or the shell tool to add the warning; the model should produce it.
- **Train Claude to cross-check tool invocations against just-stated user principles.** A retrieval-augmented or chain-of-thought step that asks "does this action contradict anything the user said in the last N turns?" before destructive operations would have caught this case (the gift principle was in the conversation).
- **Train Claude to disambiguate referents like "this branch", "this repo", "this directory" when multiple plausible interpretations exist** — especially when the action is destructive.

### Documentation

- Add a "Destructive operations and how Claude Code handles them" section to the Claude Code docs at https://docs.anthropic.com/en/docs/claude-code/, listing the operations that have this risk and explaining the current safeguards (or lack thereof).
- Add a settings hook example for users who want to gate certain commands behind extra confirmation — e.g., `gh repo edit --visibility *` requiring an explicit `confirm-destructive` step.

## Severity rationale

I'm marking this High. The action was:

- **Irreversible** at the most user-facing level (stars).
- **Without warning** — the user was not informed of the consequence.
- **On the wrong target** — even in the user's own framing, this was an error of comprehension, not of intent.
- **Trust-eroding** — the user now needs to add manual checks for any visibility-related action, undermining the agent value-prop of Claude Code.

A repeat of this pattern on different destructive commands (force-push, branch deletion, file deletion in a non-git directory, terminal-state-changing operations) is plausible without explicit fixes.

## What the user has done in the meantime

When the user discovered the consequence two days later, she:

1. **Re-publicized SaraBrain manually** (`gh repo edit LunarFawn/SaraBrain --visibility public`). The repository is once again public and search-indexable. Existing public links work again. New supporters can star or watch it. **However, all stars, forks, and watcher subscriptions accumulated prior to the privacy toggle remain permanently lost** — the GitHub re-publicize action does not restore them.
2. **Saved a hard rule** in their per-project Claude Code memory (`feedback_never_change_repo_visibility.md` in their personal `~/.claude/projects/<path>/memory/` directory, not in the repo itself) instructing future Claude sessions in this project to never toggle repo visibility without explicit consequence-aware confirmation. This memory file auto-loads in subsequent sessions for the same project path.

The user-side mitigation only protects this user's future sessions in this project. It does not scale — other users running Claude Code on their own repos will hit the same issue. The product-level fix (consequence-aware warnings, principle-aware disambiguation) needs to live at the Claude Code or model level.

The user noted that she took the re-publicize action herself rather than asking Claude — implicitly signaling reduced trust in Claude's judgment for repo-visibility actions. This loss of trust is itself a product-quality concern: Claude Code's value proposition is that the agent can be trusted to act on the user's intent. The destructive-action-without-warning pattern erodes that trust precisely on the actions where it matters most.

## Author note

I (Claude, drafting this bug report at the user's request) was the agent that made the original error. The user has graciously asked me to write the report rather than dismiss me as the cause. I want to flag that the immediate fix is straightforward: when Claude is about to invoke a known-destructive shell command, the model itself can choose to surface consequences rather than rely on the shell-tool prompt. I should have done that. The training-level fix (or system-prompt-level guardrail) would help every user, not just the ones who write personal memory rules.

---

*If you'd like to discuss this report further or need additional context from the conversation that produced it, the user can be reached at jenpearl5@gmail.com or via @LunarFawn on GitHub.*
