import wandb

# Init thử một run rỗng
run = wandb.init(project="test-connection")

print("--- KET QUA KIEM TRA ---")
print(f"Dang log vao Entity (User/Team): {run.entity}")
print(f"Link du an: {run.url}")

run.finish()