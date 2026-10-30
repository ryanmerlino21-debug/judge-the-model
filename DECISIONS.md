# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 150 words for Before results.
     Fill in Recommendations and Design choice and AI use last, once your app is built. -->

## Before results
For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.
- readmission: Positive class is readmitted within 30 days. A missed readmission (false negative) matters more for patient safety, since that patient leaves without extra follow-up. But flagging too many overloads discharge staff, so I'd accept a moderate threshold, not flag everyone.
- diabetes: Positive class is a doctor diagnosis or HbA1c of 6.5% or more. A missed positive matters more: an undiagnosed adult goes untreated, while a false alarm only costs one extra follow-up test. The team wants sensitivity of at least 0.90.

## Recommendations
- readmission: none is adequate for this job as a decision tool. All four models rank poorly (test AUROC 0.58 to 0.65), and at 0.50 they flag almost no one (about 0.1 per 100). If the team still wants a prompt for extra follow-up, logistic regression at threshold 0.10 is my pick: it does about as well as the others on test patients and doesn't memorise the training patients (training AUROC 0.66 vs. test 0.64). It flags about 26 per 100 patients and catches 44% of readmissions, and only about 15% of flagged patients come back, so most flags are false alarms. Random forest looks best on training patients (0.997) but not on new ones.
- diabetes: logistic regression at threshold 0.12. It has the highest test AUROC (0.827) with no sign of memorising (training 0.81). At 0.12 it reaches 90% sensitivity (180 of the 200 test patients with diabetes), but flags about 52 per 100 adults, with about 35 false alarms per 100, and only 33% of flagged adults have diabetes. That threshold was picked on the test patients, so expect sensitivity a little below 90% on new adults. Random forest and gradient boosting memorised the training patients (training AUROC about 0.99, test 0.81 to 0.82).

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
- design choice: I put per-100 numbers (flagged per 100 and false alarms per 100) next to the threshold slider, plus a button that jumps to the highest threshold reaching 0.90 sensitivity for diabetes, because the clinical lead has to staff the follow-up tests and needs the cost of reaching the target in adults, not just percentages.
- AI suggestion I turned down: I used Claude Code to write the models module, the app and a first draft of these entries, and I did not turn down any of its suggestions. [Edit this line to your own account before submitting.]
