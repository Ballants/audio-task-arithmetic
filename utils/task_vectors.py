import torch


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint)
                finetuned_state_dict = torch.load(finetuned_checkpoint)
                self.vector = {}
                for key in pretrained_state_dict:
                    # if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                    # continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, model, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_state_dict = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        model.load_state_dict(new_state_dict)
        return model

    def __mul__(self, scalar):
        """Multiply the task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * scalar
        return TaskVector(vector=new_vector)

    def __rmul__(self, scalar):
        """Multiply the task vector by a scalar (right multiplication)."""
        return self.__mul__(scalar)
