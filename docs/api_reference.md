# ChronoSynapse API Reference

## Quantum Neural Processor (QNP)

### `QuantumNeuralProcessor(num_qubits)`

Creates a Quantum Neural Processor.

**Arguments:**
- `num_qubits`: Number of qubits in the quantum circuit (default: 4).

### `create_quantum_circuit(data_vector)`

Generates a quantum circuit based on the input data vector.

**Arguments:**
- `data_vector`: A list or array representing classical data to encode into the quantum circuit.

**Returns**: A quantum circuit object.

### `execute(data_vector)`

Executes the quantum circuit and returns the result.

**Arguments:**
- `data_vector`: The input data vector for processing.

**Returns**: The quantum statevector resulting from the execution.

## Temporal Awareness Layer (TAL)

### `TemporalAwarenessLayer(time_steps)`

Creates a Temporal Awareness Layer with the specified time step length.

**Arguments:**
- `time_steps`: The number of previous time steps used for prediction.

### `train(data)`

Trains the model on temporal data.

**Arguments:**
- `data`: A sequence of data points representing the time series.

**Returns**: None

### `predict(current_data)`

Predicts the next value in the time series based on the provided data.

**Arguments:**
- `current_data`: A sequence of the latest data points.

**Returns**: The predicted next data point.
