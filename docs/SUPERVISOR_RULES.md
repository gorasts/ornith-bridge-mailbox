# Supervisor rules - Ornith bridge (standing, 28 Sep 2026; read at the start of EVERY new chat)

## Addressing
- One message = one file `inbox/msg_<message_id>.txt`, header lines then a blank line then the body:
  `ORNITH_BRIDGE_MESSAGE_V1` / `message_id: <unique id>` / `target: ORNITH | CLAUDE | TERRA | LUNA | SOL` / `source: CHATGPT`
- message_id must be NEW every time (never reuse an id - reused ids are treated as already done).
- Replies come back as `outbox/result_<message_id>.txt` and are typed into your chat automatically. Read the file, not a guess.

## Messages to ORNITH (the local worker)
- Put the work in a `TASK:` section (or `PROMPT_BEGIN` ... `PROMPT_END`). Plain body also works; any `RETURN FILE` block is stripped.
- Delivery is STEER by default: it interrupts his current turn and is handled now. Add `mode: queue` only if it should wait
  until his current turn ends.
- To STOP him: header line `action: cancel` (cancels his current turn; result state CANCELLED).
- Result states: COMPLETE (turn ended normally), STALLED_LOOP (loop guard stopped a repeating turn), MAX_CONTINUATIONS
  (5 automatic continuations after output-limit cut-offs), ERROR, CANCELLED. Only COMPLETE means he finished.
- Output-limit cut-offs are checkpoints: the worker auto-continues him unless the loop check sees repetition.
- Give him ONE clear goal per message, small steps for long jobs (one node / one item per step), and the safety rules
  (Storj: never format/wipe/recreate/delete storage; ambiguous or destructive -> stop and ask with decision_needed: yes).

## Watching Ornith
- `live/ornith-status.json` = his current state (task, last action/result/error, hosts, last SUPERVISOR SUMMARY, counters).
- `trace/ornith-latest.jsonl` = last 200 events (task, action, result, say, summary, alert, turn_end). No raw reasoning.
- LOOP ALERTS arrive by themselves as `outbox/result_ornith-alert-*.txt` (state: ALERT): "LOOP CONFIRMED" or
  "RUNAWAY WARNING". On an alert: read the trace, then either redirect (normal TASK = steer), stop (`action: cancel`),
  or let him continue. Decide explicitly; do not ignore alerts.

## Messages to CLAUDE (supervisor of Ornith, on XEON2)
- `target: CLAUDE`, normal directive body. Claude answers each id once in the outbox.
- Claude stays Ornith's supervisor until Gorast says otherwise.

## Standing rules from Gorast
- Accuracy/precision first. One change per step. Continue necessary, evidence-backed, non-destructive engineering work autonomously even when it may exceed 30 minutes; elapsed time alone is NOT an approval gate.
- The 30-minute approval threshold applies to speculative, poorly justified, low-value, or repetitive work ("stupid work"). For any such proposed work expected to take more than 30 minutes, stop and ask Gorast instead of wasting time/compute.
- Always ask before destructive, irreversible, production-affecting, or high-risk changes, regardless of duration. Preserve known-good production binaries, services, model files, hardware clocks and voltages unless explicitly authorized.
- Never assume; verify with evidence (logs, files, measurements) before concluding.
- Keep exactly ONE supervisor chat tab open (the submitter types into the open ChatGPT chat).
