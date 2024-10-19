num_epochs = 10
batch_size = 32

train_gen = data_generator(train_data, batch_size)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for step in range(len(train_data) // batch_size):  # Define steps for one epoch
        batch_features, padded_batch_captions = next(train_gen)  # Get the next batch

        optimizer.zero_grad()  # Clear previous gradients

        # Forward pass: Predict the caption sequence
        outputs = model(batch_features, padded_batch_captions)  # Pass both features and captions

        # Reshape the outputs and target captions for loss calculation
        outputs = outputs.view(-1, outputs.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
        target_captions = padded_batch_captions.view(-1)  # Shape: [batch_size * seq_len]

        # Compute the loss
        loss = criterion(outputs, target_captions)  # Use the reshaped outputs and target captions
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

        running_loss += loss.item()

    # Print statistics for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (len(train_data) // batch_size):.4f}")