"""Back-of-envelope phone budget. Every input is an assumption, stated here; outputs are arithmetic only."""
W_BYTES = 2.5e9          # Qwen3.5-4B at 4-bit, with overheads (moderator's estimate)
BW = 60e9                # usable phone DRAM bandwidth, bytes/s (flagship LPDDR5X peak ~77-85 GB/s)
J_DEC = 0.30             # J per decoded token, whole phone (PowerInfer-2 measured 0.26 J/token; rule of thumb 0.1 J/token/B params)
J_PRE = 0.03             # J per prefilled token (prefill is compute-bound and ~10x cheaper per token)
TOK_DEC, TOK_PRE = 20, 200   # tokens/s, the project's assumptions
BATT_J = 18 * 3600       # ~18 Wh battery

def cost(pre, dec, batch=1):
    j = pre * J_PRE + dec * J_DEC / (0.6 * batch if batch > 1 else 1)   # batching shares the weight stream; 60% efficiency
    s = pre / TOK_PRE + dec / (TOK_DEC * (0.6 * batch if batch > 1 else 1))
    return j, s

print(f"decode ceiling from bandwidth: {BW / W_BYTES:.0f} tokens/s (weights streamed once per token)")
for name, pre, dec in [("answer now (B, thinking on: ~2,000 prompt, 4,172 median completion)", 2000, 4172),
                       ("answer proposed (resolved evidence, thinking off: ~600 prompt, ~40 completion)", 600, 40),
                       ("answer proposed, engine template (query parse only: ~400 prompt, ~25 completion)", 400, 25)]:
    j, s = cost(pre, dec)
    print(f"{name}: {j:,.0f} J = {100 * j / BATT_J:.2f}% battery, {s:,.0f} s")
for name, pre, dec, b in [("write now (B writer: 139 prompt + 15 completion per message, sequential)", 139, 15, 1),
                          ("write proposed, live (70 new prompt tokens + 30 completion, cached instructions)", 70, 30, 1),
                          ("write proposed, at night, 8 messages decoded together", 70, 30, 8)]:
    j, s = cost(pre, dec, b)
    print(f"{name}: {j:.1f} J, {s:.2f} s per message; 50 messages/day = {50 * j:,.0f} J/day = {100 * 50 * j / BATT_J:.2f}% battery")
# memory side
msgs_10y = 6292 / 22 * 120
print(f"\n10 years at the test-2 density: {msgs_10y:,.0f} messages; text {msgs_10y * 55 / 1e6:.1f} MB; "
      f"descriptions {msgs_10y * 120 / 1e6:.1f} MB; binary 384-bit codes {msgs_10y * 48 / 1e6:.1f} MB")
scan_bytes = msgs_10y * 48
print(f"one full scan of the codes moves {scan_bytes / 1e6:.1f} MB ({scan_bytes / BW * 1e3:.2f} ms); one decoded token moves "
      f"{W_BYTES / 1e9:.1f} GB = {W_BYTES / scan_bytes:,.0f} full memory scans")
print(f"full re-solve, measured 0.04 s for 6,292 messages in Python on one server core -> 10 years ~"
      f"{0.04 * msgs_10y / 6292:.1f} s server, assume 3-5x slower on a phone core: ~{0.04 * msgs_10y / 6292 * 4:.0f} s")
