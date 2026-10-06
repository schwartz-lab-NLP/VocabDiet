# Method

The published paper is the reference for the experimental scope. This repository has two modeling workflows.

## Post-hoc adaptation

For a surface form `w`, the decomposition map supplies a base `b(w)` and transformations `T(w)`. Input embeddings are composed as

```
e(w) = e(b(w)) + sum(e(t) for t in T(w))
```

Output scores use the same decomposition with separately initialized unembedding vectors:

```
logit(w) = h @ u(b(w)) + sum(h @ u(t) for t in T(w))
```

A single softmax normalizes the scores over the candidate surface vocabulary. Unmapped words retain their original representation. Input and output transformations are initialized independently from mean offsets over in-vocabulary pairs demonstrating exactly one transformation.

The language-modeling experiments exclude derivations and compositions that fail the detokenization probe. Excluded words retain their original tokenization. The broader representation analysis includes derivations so their resolution can be measured.

Adaptation trains input transformations against the original model first, then output transformations against the model resulting from the first stage. The backbone and base vectors remain frozen. The lightweight adaptation also uses LoRA in the last eight layers, with rank and alpha both 256.

## Pretraining

Pretraining uses additive inputs and a factorized output:

```
p(w | history) = p(b(w) | history) * product(p(t_g | b(w), history) for g in groups)
```

Each group includes a null choice. Transformation heads receive the hidden state and a representation of the selected base. Training uses the gold base; inference selects the base before selecting transformation values. The final paper specifies the base's **unembedding** vector for conditioning. Summed joint negative log-likelihood, including all transformation groups, is used for bits-per-byte evaluation.

English starts from GPT-2's vocabulary and restricts surface output to the original vocabulary. Spanish starts from a 32k BPE vocabulary trained on 10B bytes of Spanish FineWeb-2 and permits out-of-vocabulary compositions. Both use whitespace-prefix transformations.

## Metrics

Bits per byte (lower is better) is total negative log-likelihood in nats divided by `log(2)` and the number of UTF-8 bytes. Bytes per token (higher is better) is total UTF-8 bytes divided by token count. Compute both from shared held-out text when comparing tokenizers.
