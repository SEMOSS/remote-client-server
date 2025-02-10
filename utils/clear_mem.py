import torch
import gc


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass

    torch.cuda.empty_cache()


if __name__ == "__main__":
    clear_gpu_memory()
