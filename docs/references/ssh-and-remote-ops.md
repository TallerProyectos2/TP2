# SSH And Remote Operations

## Access Path

- Run repo orchestration from an operator MacBook with Tailscale access.
- Use direct SSH from the MacBook to EPC and Jetson.
- Use EPC as the SSH proxy to the eNodeB.
- Preferred operator flow:
  - `ssh tp2@100.97.19.112`
  - `ssh grupo4@100.115.99.8`
  - `ssh -J tp2@100.97.19.112 tp2@10.10.10.2`
- The `tp2` user on the EPC is configured for key-based SSH to the `tp2` user on the eNodeB, so the second hop should not require an interactive password.
- The `ops/bin/tp2-up` launcher defaults to the same topology through `TP2_EPC_SSH`, `TP2_JETSON_SSH`, and `TP2_ENB_SSH_PROXY`.
- Keep passwords out of repository files, docs, scripts, and Jira comments. If credentials change, update the machine access method, not the secret itself.

## Safety Rules

- Inspect before modifying.
- Prefer process, port, and config inspection before service restarts.
- Avoid restarting the EPC and eNodeB in the same troubleshooting step unless the reason is explicit.
- Do not write credentials into repository files.

## Typical Safe Checks

- process status
- listening ports
- config file inspection
- interface addressing
- reachability tests

## Escalate Before

- disruptive restarts
- firewall rewrites
- deleting state
- changing working HSS or radio config without a clear rollback
