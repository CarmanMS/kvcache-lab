// A small scalar reference, independent of every production math helper.
// Weights and embeddings are test inputs. Fixed dimensions make each equation
// inspectable: residual width 2, two Q heads, one KV head, head dimension 2.
export function twoDimensionalOracle(inputs, weights) {
  const normalize = ([x, y]) => {
    const denominator = Math.sqrt((x * x + y * y) / 2 + 0.000001);
    return [x / denominator, y / denominator];
  };
  const project = ([x, y], matrix) => matrix[0].map((_, j) => x * matrix[0][j] + y * matrix[1][j]);
  const rotate = ([x, y], position) => [
    x * Math.cos(position) - y * Math.sin(position),
    x * Math.sin(position) + y * Math.cos(position),
  ];
  let states = inputs.map((input) => [...input]);
  const traces = inputs.map(() => []);
  for (const [layer, w] of weights.entries()) {
    const projections = states.map((input, t) => {
      const normalized = normalize(input);
      const qFlat = project(normalized, w.wQ);
      return {
        normalized,
        q: [rotate(normalize(qFlat.slice(0, 2)), t), rotate(normalize(qFlat.slice(2, 4)), t)],
        k: [rotate(normalize(project(normalized, w.wK)), t)],
        v: [project(normalized, w.wV)],
      };
    });
    states = states.map((input, t) => {
      const heads = projections[t].q.map((query) => {
        const scores = [],
          weights = [],
          context = [0, 0];
        let denominator = 0;
        for (let position = 0; position <= t; position++) {
          const [kx, ky] = projections[position].k[0];
          const score = (query[0] * kx + query[1] * ky) / Math.sqrt(2);
          scores.push(score);
          denominator += Math.exp(score);
        }
        for (let position = 0; position <= t; position++) {
          const weight = Math.exp(scores[position]) / denominator;
          weights.push(weight);
          context[0] += weight * projections[position].v[0][0];
          context[1] += weight * projections[position].v[0][1];
        }
        return { scores, weights, context };
      });
      const contexts = [...heads[0].context, ...heads[1].context];
      const attentionOutput = [0, 0];
      for (let i = 0; i < 4; i++) {
        attentionOutput[0] += contexts[i] * w.wO[i][0];
        attentionOutput[1] += contexts[i] * w.wO[i][1];
      }
      const residual = [input[0] + attentionOutput[0], input[1] + attentionOutput[1]];
      const ffnInput = normalize(residual);
      const gate = project(ffnInput, w.wGate),
        up = project(ffnInput, w.wUp);
      const output = [...residual];
      for (let i = 0; i < 4; i++) {
        const activation = (gate[i] * up[i]) / (1 + Math.exp(-gate[i]));
        output[0] += activation * w.wDown[i][0];
        output[1] += activation * w.wDown[i][1];
      }
      traces[t][layer] = {
        input,
        ...projections[t],
        heads,
        attentionOutput,
        residual,
        ffnInput,
        output,
      };
      return output;
    });
  }
  return { outputs: states, traces };
}
