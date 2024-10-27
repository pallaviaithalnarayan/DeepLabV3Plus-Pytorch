import matplotlib.pyplot as plt

# Step 1: Initialize an empty list to store the average loss per epoch
epoch_losses = []

# Your training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass, compute loss, backward pass, and optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track the running loss for each batch
        running_loss += loss.item()
    
    # Step 2: Calculate and store the average loss for this epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Step 3: Plot and save the loss graph after training
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='b', label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.savefig("loss_per_epoch.png")  # Save the figure
plt.show()  # Display the figure
