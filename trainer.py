import torch
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Callable, Any, Tuple


# ===================== SAM Optimizer =====================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.
    
    SAM seeks parameters that lie in neighborhoods having uniformly low loss,
    which improves model generalization.
    
    Reference: https://arxiv.org/abs/2010.01412
    
    Args:
        base_optimizer: The base optimizer (e.g., SGD, Adam)
        rho: Neighborhood size for perturbation (default: 0.05)
        adaptive: Whether to use adaptive SAM (default: False)
    """
    def __init__(self, params, base_optimizer: torch.optim.Optimizer, rho: float = 0.05, adaptive: bool = False):
        assert rho >= 0.0, f"Invalid rho value: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Compute and apply the perturbation (first forward-backward pass)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to the local maximum (w + epsilon)
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Apply the gradient at perturbed point (second forward-backward pass)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Restore original weights
        
        self.base_optimizer.step()  # Do the actual update
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Standard optimizer step (for compatibility)."""
        if closure is not None:
            with torch.enable_grad():
                closure()
        self.base_optimizer.step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ===================== Mixup & CutMix =====================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation to a batch of data.
    
    Args:
        x: Input images (B, C, H, W)
        y: Target labels (B,)
        alpha: Mixup interpolation strength (default: 1.0)
    
    Returns:
        mixed_x: Mixed images
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation to a batch of data.
    
    Args:
        x: Input images (B, C, H, W)
        y: Target labels (B,)
        alpha: CutMix interpolation strength (default: 1.0)
    
    Returns:
        mixed_x: CutMix images
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient (area ratio)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    _, _, H, W = x.shape
    
    # Get random box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute mixed loss for Mixup/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===================== Adaptive Training Config =====================

class AdaptiveTrainingConfig:
    """
    Configuration for adaptive training that monitors generalization gap
    and adjusts training parameters accordingly.
    
    As per plan.md Section 4.3:
    - Case 1: Overfitting (G increases) → Increase Weight Decay 1.5x, Increase CutMix probability
    - Case 2: Underfitting (Loss_train decreases slowly) → Reduce Augmentation
    - Case 3: Plateau (Loss goes flat) → Activate SAM for next epochs
    """
    def __init__(
        self,
        enabled: bool = True,
        check_interval: int = 5,  # Check every N epochs
        gap_threshold: float = 0.1,  # Threshold for significant gap change
        plateau_threshold: float = 0.01,  # Loss change threshold for plateau
        max_weight_decay_factor: float = 10.0,  # Maximum WD multiplier
        sam_epochs_on_plateau: int = 10,  # Epochs to use SAM after plateau
    ):
        self.enabled = enabled
        self.check_interval = check_interval
        self.gap_threshold = gap_threshold
        self.plateau_threshold = plateau_threshold
        self.max_weight_decay_factor = max_weight_decay_factor
        self.sam_epochs_on_plateau = sam_epochs_on_plateau
        
        # State tracking
        self.current_wd_factor = 1.0
        self.mixup_enabled = True
        self.cutmix_enabled = True
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
        self.cutmix_prob = 0.5
        self.sam_active = False
        self.sam_remaining_epochs = 0
        self.previous_gaps = []
        self.previous_train_losses = []


class LabelSmoothingCrossEntropy(torch.nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        weight: Class weights for imbalanced datasets
    """
    def __init__(self, smoothing: float = 0.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing == 0.0:
            # Standard cross entropy
            if self.weight is not None:
                return torch.nn.functional.cross_entropy(pred, target, weight=self.weight)
            return torch.nn.functional.cross_entropy(pred, target)
        
        n_classes = pred.size(-1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(pred.device)
            sample_weights = weight[target]
            loss = (-true_dist * log_preds).sum(dim=-1) * sample_weights
        else:
            loss = (-true_dist * log_preds).sum(dim=-1)
        
        return loss.mean()


def count_images_per_class(dataloader):
    class_counts = {}
    
    class_to_idx = dataloader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for class_name, idx in class_to_idx.items():
        class_counts[idx] = 0
    for _, labels in tqdm(dataloader, desc = "Counting images per class"):
        for label in labels:
            class_idx = label.item()
            class_counts[class_idx] += 1
    return class_counts

def calculate_class_weights(class_counts, weight_type = 'inverse'):
    counts = []
    class_indices = []
    for class_idx, count in class_counts.items():
        counts.append(count)
        class_indices.append(class_idx)
    
    sorted_indices = sorted(zip(class_indices, counts))
    class_indices = [pair[0] for pair in sorted_indices]
    counts = [pair[1] for pair in sorted_indices]
    counts = np.array(counts)
    if weight_type == 'inverse':
        weights = 1.0 / counts
    elif weight_type == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError("Invalid weight type. Choose 'inverse' or 'sqrt_inverse'.")
    weights = weights / np.sum(weights) * len(class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float)
    return class_weights
    

class Trainer:
    def __init__(self, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, 
                 device=None, num_epochs=25, save_path=None, wandb_run=None, tb_writer=None,
                 optuna_trial=None, optuna_prune_metric: str = 'val_acc',
                 use_mixup: bool = False, mixup_alpha: float = 0.2,
                 use_cutmix: bool = False, cutmix_alpha: float = 1.0, cutmix_prob: float = 0.5,
                 use_sam: bool = False, sam_rho: float = 0.05,
                 adaptive_config: Optional[AdaptiveTrainingConfig] = None):
        """
        Trainer class for training PyTorch models with advanced training techniques.
        
        Args:
            model: PyTorch model to train
            dataloaders: Dict with 'train' and 'test' dataloaders
            dataset_sizes: Dict with 'train' and 'test' dataset sizes
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on (optional, defaults to CUDA if available)
            num_epochs: Number of epochs to train
            save_path: Path to save best model (optional)
            wandb_run: W&B run object for logging (optional)
            tb_writer: TensorBoard writer (optional)
            optuna_trial: Optuna trial object for HPO pruning (optional)
            optuna_prune_metric: Metric to use for Optuna pruning ('val_acc' or 'val_loss')
            use_mixup: Whether to use Mixup augmentation (default: False)
            mixup_alpha: Mixup interpolation strength (default: 0.2)
            use_cutmix: Whether to use CutMix augmentation (default: False)
            cutmix_alpha: CutMix interpolation strength (default: 1.0)
            cutmix_prob: Probability of applying CutMix vs Mixup (default: 0.5)
            use_sam: Whether to use SAM optimizer (default: False)
            sam_rho: SAM perturbation radius (default: 0.05)
            adaptive_config: Adaptive training configuration (optional)
        """
        super().__init__()
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.wandb_run = wandb_run
        self.tb_writer = tb_writer
        self.optuna_trial = optuna_trial
        self.optuna_prune_metric = optuna_prune_metric
        
        # Mixup/CutMix settings
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        
        # SAM optimizer settings
        self.use_sam = use_sam
        self.sam_rho = sam_rho
        if use_sam:
            # Wrap optimizer with SAM
            self.sam_optimizer = SAM(model.parameters(), optimizer, rho=sam_rho)
        else:
            self.sam_optimizer = None
        
        # Adaptive training settings
        self.adaptive_config = adaptive_config if adaptive_config else AdaptiveTrainingConfig(enabled=False)
        
        # Lưu trạng thái tốt nhất
        self.best_model = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.model = model.to(self.device)
    
    def train(self):
        since = time.time()
        
        # Watch model gradients/parameters if using W&B
        if self.wandb_run is not None:
            try:
                self.wandb_run.watch(self.model, log="gradients", log_freq=100)
            except Exception:
                pass
                
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            epoch_start = time.time()
            
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                
                dataloader = self.dataloaders[phase]
                # Tqdm bar setup
                progress_bar = tqdm(dataloader, desc=f"{phase} epoch {epoch+1}/{self.num_epochs}", unit="batch")
                
                seen_samples = 0
                
                # --- BẮT ĐẦU VÒNG LẶP BATCH ---
                for inputs, labels in progress_bar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Apply Mixup/CutMix in training phase
                    use_mixup_this_batch = False
                    use_cutmix_this_batch = False
                    if phase == 'train' and (self.use_mixup or self.use_cutmix):
                        r = np.random.rand()
                        if self.use_cutmix and self.use_mixup:
                            # Both enabled: use cutmix_prob to decide
                            if r < self.cutmix_prob:
                                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, self.cutmix_alpha)
                                use_cutmix_this_batch = True
                            else:
                                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, self.mixup_alpha)
                                use_mixup_this_batch = True
                        elif self.use_cutmix:
                            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, self.cutmix_alpha)
                            use_cutmix_this_batch = True
                        elif self.use_mixup:
                            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, self.mixup_alpha)
                            use_mixup_this_batch = True
                    
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        
                        # Compute loss (with Mixup/CutMix if applied)
                        if phase == 'train' and (use_mixup_this_batch or use_cutmix_this_batch):
                            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                            # For accuracy, use original labels (approximation)
                            _, preds = torch.max(outputs, 1)
                        else:
                            loss = self.criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                        
                        # Backward + Optimize
                        if phase == 'train':
                            if self.use_sam and self.sam_optimizer is not None:
                                # SAM: Two-step optimization
                                loss.backward()
                                self.sam_optimizer.first_step(zero_grad=True)
                                
                                # Second forward-backward pass
                                outputs_2 = self.model(inputs)
                                if use_mixup_this_batch or use_cutmix_this_batch:
                                    loss_2 = mixup_criterion(self.criterion, outputs_2, targets_a, targets_b, lam)
                                else:
                                    loss_2 = self.criterion(outputs_2, labels)
                                loss_2.backward()
                                self.sam_optimizer.second_step(zero_grad=True)
                            else:
                                # Standard optimization
                                loss.backward()
                                self.optimizer.step()
                            
                            # [FIX 2] STEP SCHEDULER THEO BATCH (QUAN TRỌNG NHẤT)
                            # Dành cho Cosine, Linear Warmup, StepLR...
                            if self.scheduler is not None:
                                # Trừ trường hợp dùng ReduceLROnPlateau (loại này step theo epoch)
                                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.scheduler.step()
                            
                            # Log batch metric cho W&B để vẽ chart mượt
                            if self.wandb_run is not None and seen_samples % (10 * inputs.size(0)) == 0:
                                try:
                                    self.wandb_run.log({
                                        "train/batch_loss": loss.item(),
                                        "lr": self.optimizer.param_groups[0]['lr']
                                    })
                                except Exception:
                                    pass
                        
                        # Stats accumulation
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        seen_samples += inputs.size(0)
                        current_loss = running_loss / seen_samples
                        progress_bar.set_postfix(loss=f"{current_loss:.4f}")
                
                # --- KẾT THÚC VÒNG LẶP BATCH ---
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                # Update history
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.item())
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.item())
                
                # Log epoch metrics to W&B
                if self.wandb_run is not None:
                    try:
                        log_payload = {
                            f'{phase}/loss': float(epoch_loss),
                            f'{phase}/acc': float(epoch_acc),
                            'epoch': epoch + 1,
                        }
                        # Log LR cuối epoch để tham khảo thêm
                        if phase == 'train' and len(self.optimizer.param_groups) > 0:
                            log_payload['lr_epoch_end'] = float(self.optimizer.param_groups[0].get('lr', 0.0))
                        
                        self.wandb_run.log(log_payload)
                    except Exception:
                        pass
                
                # Log epoch metrics to TensorBoard
                if self.tb_writer is not None:
                    try:
                        self.tb_writer.add_scalar(f'{phase}/loss', float(epoch_loss), epoch + 1)
                        self.tb_writer.add_scalar(f'{phase}/acc', float(epoch_acc), epoch + 1)
                        if phase == 'train' and len(self.optimizer.param_groups) > 0:
                            self.tb_writer.add_scalar('lr', float(self.optimizer.param_groups[0].get('lr', 0.0)), epoch + 1)
                    except Exception:
                        pass
                    
                # [FIX 3] LOGIC SAVE MODEL CHUẨN
                if phase == 'test':
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    
                    # Ưu tiên lưu theo Accuracy (Quan trọng nhất với bài toán phân loại)
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        self.best_model = copy.deepcopy(self.model.state_dict())
                        
                        # Lưu file ngay lập tức khi có Acc kỷ lục mới
                        if self.save_path:
                            torch.save(self.model.state_dict(), self.save_path)
                            print(f"--> Model saved to {self.save_path} (New Best Acc: {self.best_acc:.4f})")
                            
                            # Log best checkpoint to wandb
                            if self.wandb_run is not None:
                                try:
                                    self.wandb_run.log({'best/val_acc': float(self.best_acc)})
                                except Exception:
                                    pass

                    # Theo dõi thêm Loss, nhưng không save đè lên file Acc
                    if epoch_loss < self.best_val_loss:
                        self.best_val_loss = epoch_loss

            # [FIX 4] STEP SCHEDULER THEO EPOCH (Chỉ dành cho ReduceLROnPlateau)
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Plateau cần metric val_loss để quyết định giảm LR
                self.scheduler.step(self.history['val_loss'][-1])
            
            # Optuna pruning: report intermediate value and check if trial should be pruned
            if self.optuna_trial is not None:
                try:
                    import optuna
                    # Report validation metric to Optuna
                    if self.optuna_prune_metric == 'val_acc':
                        prune_value = self.history['val_acc'][-1]
                    else:
                        # For loss, negate so higher is better (Optuna maximizes by default for pruning)
                        prune_value = -self.history['val_loss'][-1]
                    
                    self.optuna_trial.report(prune_value, epoch)
                    
                    # Check if trial should be pruned
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()
                except ImportError:
                    pass  # Optuna not installed
            
            # ========== ADAPTIVE TRAINING LOGIC ==========
            # Check every N epochs and adjust training parameters based on generalization gap
            if self.adaptive_config.enabled and (epoch + 1) % self.adaptive_config.check_interval == 0:
                self._apply_adaptive_training(epoch)
                        
            epoch_end = time.time()
            print(f'Epoch {epoch+1} completed in {epoch_end - epoch_start:.0f} seconds')
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best test Acc: {self.best_acc:.4f}')
        print(f'Best test Loss: {self.best_val_loss:.4f}')
        
        # Final summary log
        if self.wandb_run is not None:
            try:
                self.wandb_run.summary['best_val_acc'] = float(self.best_acc)
                self.wandb_run.summary['best_val_loss'] = float(self.best_val_loss)
            except Exception:
                pass
        
        # Load best model weights before returning
        self.model.load_state_dict(self.best_model)
        return self.model, self.history
    
    def _apply_adaptive_training(self, epoch: int):
        """
        Apply adaptive training logic based on generalization gap analysis.
        
        As per plan.md Section 4.3:
        - Case 1: Overfitting (G increases rapidly) → Increase Weight Decay 1.5x, Increase CutMix probability
        - Case 2: Underfitting (train loss decreases slowly) → Reduce Augmentation, consider Cyclical LR
        - Case 3: Plateau (Loss goes flat) → Activate SAM for next epochs
        """
        cfg = self.adaptive_config
        
        # Calculate current generalization gap
        if len(self.history['val_loss']) < 2 or len(self.history['train_loss']) < 2:
            return
        
        current_gap = self.history['val_loss'][-1] - self.history['train_loss'][-1]
        previous_gap = self.history['val_loss'][-2] - self.history['train_loss'][-2]
        gap_change = current_gap - previous_gap
        
        train_loss_change = self.history['train_loss'][-2] - self.history['train_loss'][-1]
        val_loss_change = self.history['val_loss'][-2] - self.history['val_loss'][-1]
        
        cfg.previous_gaps.append(current_gap)
        cfg.previous_train_losses.append(self.history['train_loss'][-1])
        
        # Case 1: Overfitting - Gap increasing rapidly
        if gap_change > cfg.gap_threshold:
            print(f"\n[ADAPTIVE] Overfitting detected (gap increased by {gap_change:.4f})")
            
            # Increase weight decay
            if cfg.current_wd_factor < cfg.max_weight_decay_factor:
                cfg.current_wd_factor *= 1.5
                for param_group in self.optimizer.param_groups:
                    if 'weight_decay' in param_group:
                        param_group['weight_decay'] *= 1.5
                print(f"  -> Increased weight decay by 1.5x (factor: {cfg.current_wd_factor:.2f})")
            
            # Increase CutMix probability
            if self.use_cutmix:
                self.cutmix_prob = min(0.9, self.cutmix_prob + 0.1)
                print(f"  -> Increased CutMix probability to {self.cutmix_prob:.2f}")
        
        # Case 2: Underfitting - Training loss not decreasing
        elif train_loss_change < cfg.plateau_threshold and current_gap < 0:
            print(f"\n[ADAPTIVE] Underfitting detected (train loss barely decreasing)")
            
            # Reduce augmentation intensity
            if self.use_mixup:
                self.mixup_alpha = max(0.1, self.mixup_alpha * 0.8)
                print(f"  -> Reduced Mixup alpha to {self.mixup_alpha:.2f}")
            if self.use_cutmix:
                self.cutmix_prob = max(0.2, self.cutmix_prob - 0.1)
                print(f"  -> Reduced CutMix probability to {self.cutmix_prob:.2f}")
        
        # Case 3: Plateau - Both train and val loss not changing much
        elif abs(train_loss_change) < cfg.plateau_threshold and abs(val_loss_change) < cfg.plateau_threshold:
            print(f"\n[ADAPTIVE] Plateau detected (losses not changing)")
            
            # Activate SAM if not already using
            if not self.use_sam and cfg.sam_remaining_epochs == 0:
                cfg.sam_remaining_epochs = cfg.sam_epochs_on_plateau
                print(f"  -> Activating SAM for next {cfg.sam_epochs_on_plateau} epochs")
                # Create SAM wrapper
                self.sam_optimizer = SAM(self.model.parameters(), self.optimizer, rho=self.sam_rho)
                self.use_sam = True
        
        # Check if SAM should be deactivated
        if cfg.sam_remaining_epochs > 0:
            cfg.sam_remaining_epochs -= 1
            if cfg.sam_remaining_epochs == 0 and not self.use_sam:  # Was temporarily enabled
                self.use_sam = False
                self.sam_optimizer = None
                print(f"\n[ADAPTIVE] Deactivating temporary SAM")
    
    def fasttrain(self):
        self.scaler = GradScaler()
        since = time.time()
        
        # Watch model gradients/parameters if using W&B
        if self.wandb_run is not None:
            try:
                self.wandb_run.watch(self.model, log="gradients", log_freq=100)
            except Exception:
                pass
                
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            epoch_start = time.time()
            
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                
                dataloader = self.dataloaders[phase]
                # Tqdm bar setup
                progress_bar = tqdm(dataloader, desc=f"{phase} epoch {epoch+1}/{self.num_epochs}", unit="batch")
                
                seen_samples = 0
                
                # --- BẮT ĐẦU VÒNG LẶP BATCH ---
                for inputs, labels in progress_bar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Forward
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        
                    # Backward + Optimize
                    if phase == 'train':
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # [FIX 2] STEP SCHEDULER THEO BATCH (QUAN TRỌNG NHẤT)
                        # Dành cho Cosine, Linear Warmup, StepLR...
                        if self.scheduler is not None:
                            # Trừ trường hợp dùng ReduceLROnPlateau (loại này step theo epoch)
                            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.scheduler.step()
                        
                        # Log batch metric cho W&B để vẽ chart mượt
                        if self.wandb_run is not None and seen_samples % (10 * inputs.size(0)) == 0:
                            try:
                                self.wandb_run.log({
                                    "train/batch_loss": loss.item(),
                                    "lr": self.optimizer.param_groups[0]['lr']
                                })
                            except Exception:
                                pass
                    
                    # Stats accumulation
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    seen_samples += inputs.size(0)
                    current_loss = running_loss / seen_samples
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}")
                
                # --- KẾT THÚC VÒNG LẶP BATCH ---
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                # Update history
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.item())
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.item())
                
                # Log epoch metrics to W&B
                if self.wandb_run is not None:
                    try:
                        log_payload = {
                            f'{phase}/loss': float(epoch_loss),
                            f'{phase}/acc': float(epoch_acc),
                            'epoch': epoch + 1,
                        }
                        # Log LR cuối epoch để tham khảo thêm
                        if phase == 'train' and len(self.optimizer.param_groups) > 0:
                            log_payload['lr_epoch_end'] = float(self.optimizer.param_groups[0].get('lr', 0.0))
                        
                        self.wandb_run.log(log_payload)
                    except Exception:
                        pass
                
                # Log epoch metrics to TensorBoard
                if self.tb_writer is not None:
                    try:
                        self.tb_writer.add_scalar(f'{phase}/loss', float(epoch_loss), epoch + 1)
                        self.tb_writer.add_scalar(f'{phase}/acc', float(epoch_acc), epoch + 1)
                        if phase == 'train' and len(self.optimizer.param_groups) > 0:
                            self.tb_writer.add_scalar('lr', float(self.optimizer.param_groups[0].get('lr', 0.0)), epoch + 1)
                    except Exception:
                        pass
                    
                # [FIX 3] LOGIC SAVE MODEL CHUẨN
                if phase == 'test':
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    
                    # Ưu tiên lưu theo Accuracy (Quan trọng nhất với bài toán phân loại)
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        self.best_model = copy.deepcopy(self.model.state_dict())
                        
                        # Lưu file ngay lập tức khi có Acc kỷ lục mới
                        if self.save_path:
                            torch.save(self.model.state_dict(), self.save_path)
                            print(f"--> Model saved to {self.save_path} (New Best Acc: {self.best_acc:.4f})")
                            
                            # Log best checkpoint to wandb
                            if self.wandb_run is not None:
                                try:
                                    self.wandb_run.log({'best/val_acc': float(self.best_acc)})
                                except Exception:
                                    pass

                    # Theo dõi thêm Loss, nhưng không save đè lên file Acc
                    if epoch_loss < self.best_val_loss:
                        self.best_val_loss = epoch_loss

            # [FIX 4] STEP SCHEDULER THEO EPOCH (Chỉ dành cho ReduceLROnPlateau)
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Plateau cần metric val_loss để quyết định giảm LR
                self.scheduler.step(self.history['val_loss'][-1])
            
            # Optuna pruning: report intermediate value and check if trial should be pruned
            if self.optuna_trial is not None:
                try:
                    import optuna
                    # Report validation metric to Optuna
                    if self.optuna_prune_metric == 'val_acc':
                        prune_value = self.history['val_acc'][-1]
                    else:
                        # For loss, negate so higher is better (Optuna maximizes by default for pruning)
                        prune_value = -self.history['val_loss'][-1]
                    
                    self.optuna_trial.report(prune_value, epoch)
                    
                    # Check if trial should be pruned
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()
                except ImportError:
                    pass  # Optuna not installed
                        
            epoch_end = time.time()
            print(f'Epoch {epoch+1} completed in {epoch_end - epoch_start:.0f} seconds')
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best test Acc: {self.best_acc:.4f}')
        print(f'Best test Loss: {self.best_val_loss:.4f}')
        
        # Final summary log
        if self.wandb_run is not None:
            try:
                self.wandb_run.summary['best_val_acc'] = float(self.best_acc)
                self.wandb_run.summary['best_val_loss'] = float(self.best_val_loss)
            except Exception:
                pass
        
        # Load best model weights before returning
        self.model.load_state_dict(self.best_model)
        return self.model, self.history
    
    def plot_history(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_history(self, filepath):
        np.savez(filepath, 
                 train_loss = np.array(self.history['train_loss']),
                 val_loss = np.array(self.history['val_loss']),
                 train_acc = np.array(self.history['train_acc']),
                 val_acc = np.array(self.history['val_acc']))
        print(f"Training history saved to {filepath}")

    def save_plot_image(self, filepath):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"Training history plot saved to {filepath}")
                      
                            
