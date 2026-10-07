Here's the update flow I have present and discuss with professor: 
One primary cause: term, category, or sentiment
find :Rare vs unseen labels; NULL prevalence; hardest domain

Class also imbalance : Positive , Negative, Neutral , Conflict , can check if we can remove Conflict
also check for : Category-label mismatch can erase otherwise useful cross-domain predictions under exact-triplet evaluation
can we study to add evaluation by not overall match ??? 
Feedback from Professor : 
1.
Use both cross-entropy and inverse-frequency weighted cross-entropy, as standard cross-entropy can still provide useful training signals.
  2.
Try Focal loss, which is a commonly used method for handling class imbalance.
  3.
Carefully tune the loss weights. For example, if the loss is:
     *
L = w1 * L(cross-entropy) + w2 * L(inverse-frequency weighted cross-entropy),
  4.
they could try different settings such as w1 = 1.0, w2 = 0.5; w1 = 0.5, w2 = 0.5; etc., and run a small parameter sweep.

Hope these tips are helpful.


Your tasks is to anlays what's to do next. not to do it all in once, anlays next step to do and fix the current protocols
Updte code with clean and easy to understand , add seperate file if possible so can call to reduce it.
My research is to bult report taxonomy are the most important which parts are the hardest ? split to calculate F1 is better than ??
