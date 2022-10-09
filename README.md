# Attempting to reverse engineer one-layer transformers

I recently interested became in Anthropic's [work] (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) on interpretability. Particularly  the sudden emergence of induction heads when you go from 1 to 2-layer transformers. With this in mind, I created a task which can be fully solved by a 2-layer transformer, but not by a 1-layer transformer. 

## The task

Predict a sequence of 24 letters where the sequence is made up of 4 different blocks of 6 letters. 
Example: ABCDEFABCDEFABCDEFABCDEF
