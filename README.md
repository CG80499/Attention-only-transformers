# Attempting to reverse engineer one-layer transformers

I recently interested became in Anthropic's [work](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) on interpretability. Particularly the sudden emergence of induction heads when you go from 1 to 2-layer transformers. 

## Task

Predict a sequence of 24 letters where the sequence is made up of 4 different blocks of 6 letters. 
Example: ABCDEFABCDEFABCDEFABCDEF

As a reminder, induction heads find the last occurrence of the current token and then predict that the pattern occurs again. So induction should be able to solve this task. However, in 1-layer transformers the key matrix is linearly dependent on the input matrix hence the model can't find the last occurrence of the current token.

##  Model

I used a simplified version of the decoder-only transformer architecture.

Simplifications:
- No layer normalization
- No MLP blocks
- Sinusoidal positional encoding is added to Q and K before the attention is computed(to avoid adding it to the residual stream)

Hyperparameters:
- 1-layer
- 4-head
- 24-token sequence
- Inner dimension: 32
- Learning rate: 0.001
- Weight decay: 1.0
- Steps: 10000
- Batch size: 256
- Total params: 26x32 + 4x32^2 + 26x32 = 5760

The model was trained on randomly generated data.

## Results

### 1-layer model without smeared keys

Below are some examples of model completions. 

Prompt: ABCDEF<br />
Completion: ECECECECECECECECEC<br />
Correct completion: ABCDEFABCDEFABCDEF<br />
<br />
Prompt: ABCDEFABCDEF<br />
Completion: CFCFCFCFCFCF<br />
Correct completion: ABCDEFABCDEF<br />
<br />
Prompt: XYZRSTXYZRST<br />
Completion: TSTSZSTSTSZST<br />
Correct completion: XYZRSTXYZRST<br />
<br />
Out of distribution example.<br />
<br />
Prompt: ABCDEFXYZRST<br />
Completion: FEFEFEFEFEFE<br />
Correct completion: -<br />
<br />
Prompt: QWERTYXYZRST<br />
Completion: YTYTYTYTYTYT<br />
Correct completion: -<br />

# Observations
The model:
 - Clearly fails at this task
 - Never repeats the same token twice in a row
 - Copies seemingly random tokens from the prompt even in out of distribution examples

# Analysis

As discussed above, the one-layer model fails because it can't use K-composition to find the last occurrence of the current token. 

Let's try looking at the eigenvalues of the OV circuits in heads 1, 2, 3 and 4. Positive eigenvalues indicate copying behaviour. Recall that k is an eigenvalue of M if and only if Mv = kv for some vector v. If k > 0 this implies that the probability of the token(or set of tokens) has increased. Note tokens are initially encoded as one-hot vectors\*.

To measure how positive the eigenvalues we will look at the sum k/|k| 
Head 1: 0.6860927<br />
Head 2: 0.85695416<br />
Head 3: 0.98481447<br />
Head 4: 0.5779195<br />

This indicates that head 3 is almost entirely dedicated to copying behaviour and provides some evidence for the second observation. The other heads also have some copying behaviour but it is less convincing.
Unfortunately, the attention mechanism is non-linear so don't have the same mathematical tools at our disposal so the evidence is weaker. Let's take a look avergae attention weights for heads 1, 2, 3 and 4 in random sample of data.

Head 1:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head1.png)<br />
Head 2:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head2.png)<br />
Head 3:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head3.png)<br />
Head 4:<br />
![image](https://github.com/CG80499/interpretability_one_layer_transformers/blob/master/images/images_head4.png)<br />


\* Why are all the matrices multiplied in the wrong order? Because the weights are transposed in the code.
Note that (AB)^T = (B^T)(A^T) and eigenvalues do not change under transposition.