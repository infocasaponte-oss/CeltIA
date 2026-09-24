# CDE shadow evaluation and promotion gate

CDE routing starts in shadow mode. The production heuristic remains authoritative while CeltIA records only route names, confidence, abstention and agreement. User prompt text is not stored in the decision-shadow table.

The admin report exposes sample count, agreement, abstention, mean confidence and common disagreement pairs.

Promotion to controlled routing must not be based on agreement alone. A labeled evaluation set is required. The offline evaluation helper compares both the existing heuristic and CDE against expected routes and applies explicit minimum sample, coverage, labeled-sample and accuracy-delta gates.

Default gate:
- at least 200 total samples;
- at least 80% CDE decision coverage;
- at least 50 labeled samples;
- CDE labeled accuracy at least 2 percentage points above the heuristic.

Passing the gate means eligible for a controlled experiment, not automatic activation. Tool authorization and security policy remain independent of routing.
