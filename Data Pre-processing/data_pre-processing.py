class CheXpertDataset(Dataset):
        def __init__(self, dataframe, transform=None, image_root=None):
            self.dataframe = dataframe
            self.transform = transform
            self.image_root = image_root
            self.label_cols = [
                'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
                'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
                'Pleural Other', 'Fracture', 'Support Devices'
            ]
            
            # Pre-validate dataset to identify problematic indices
            self.valid_indices = []
            self._validate_dataset()
        
        def _validate_dataset(self):
            """Pre-validate all images to identify valid indices"""
            print("Validating dataset images...")
            for idx in range(len(self.dataframe)):
                item = self.dataframe.iloc[idx]
                img_path = os.path.join(self.image_root, 
                                    item['Path'].replace("CheXpert-v1.0/train/", ""))
                
                if os.path.exists(img_path):
                    try:
                        # Quick validation - just try to open without loading
                        with Image.open(img_path) as img:
                            img.verify()  # Verify it's a valid image
                        self.valid_indices.append(idx)
                    except (FileNotFoundError, IOError, UnidentifiedImageError) as e:
                        logger.warning(f"Invalid image at index {idx}: {img_path} - {e}")
                else:
                    logger.warning(f"Image not found at index {idx}: {img_path}")
            
            print(f"Dataset validation complete: {len(self.valid_indices)}/{len(self.dataframe)} valid images")
            
            if len(self.valid_indices) == 0:
                raise ValueError("No valid images found in dataset!")
        
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            # Use only valid indices
            actual_idx = self.valid_indices[idx]
            item = self.dataframe.iloc[actual_idx]  # ✅ FIXED: iloc for position-based indexing

            img_path = os.path.join(
                self.image_root, item['Path'].replace("CheXpert-v1.0/train/", "")
            )
            
            try:
                # Load image
                image = Image.open(img_path).convert("RGB")
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                # Get labels
                label = item[self.label_cols].values.astype(np.float32)
                
                # Convert to tensors if needed
                if not isinstance(image, torch.Tensor):
                    image = torch.as_tensor(image)
                
                label = torch.as_tensor(label)
                
                # Return dummy text (empty) since report generation removed
                dummy_text = torch.zeros(1)
                
                return image, label, dummy_text
                
            except Exception as e:
                print(f"Unexpected error loading image at index {actual_idx}: {img_path} - {e}")
                return self._get_fallback_item()
        
        def _get_fallback_item(self):
            """Create a fallback item when image loading fails unexpectedly"""
            # Create a black image as fallback
            if hasattr(self.transform, 'transforms'):
                # Try to infer expected image size from transforms
                for t in self.transform.transforms:
                    if hasattr(t, 'size'):
                        if isinstance(t.size, (list, tuple)):
                            height, width = t.size
                        else:
                            height = width = t.size
                        break
                else:
                    height, width = 224, 224  # Default size
            else:
                height, width = 224, 224
            
            # Create fallback image
            fallback_image = torch.zeros(3, height, width)  # RGB image
            fallback_label = torch.zeros(len(self.label_cols))  # All negative labels
            dummy_text = torch.zeros(1)
            
            logger.warning("Using fallback item due to image loading failure")
            return fallback_image, fallback_label, dummy_text


    def collate_fn(batch):
        valid_batch = [
            item for item in batch
            if item is not None and item[0] is not None and item[1] is not None
        ]
        
        if len(valid_batch) == 0:
            batch_size = len(batch)
            fallback_images = torch.zeros(batch_size, 3, 224, 224)
            fallback_labels = torch.zeros(batch_size, 14)
            return fallback_images, fallback_labels, [None] * batch_size

        try:
            images, labels, texts = zip(*valid_batch)
            images = torch.stack([torch.as_tensor(img) for img in images])
            labels = torch.stack([torch.as_tensor(lbl) for lbl in labels])
            texts = list(texts)
            return images, labels, texts
        except Exception:
            batch_size = len(valid_batch)
            fallback_images = torch.zeros(batch_size, 3, 224, 224)
            fallback_labels = torch.zeros(batch_size, 14)
            return fallback_images, fallback_labels, [None] * batch_size



    # Alternative simpler collate function if you prefer
    def simple_collate_fn(batch):
        """Simpler version that just filters None and uses default collation"""
        # Remove None items
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            return None  # This will be caught in your training loop
        
        # Use default collation for the rest
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

    # Create Dataset objects
    train_dataset = CheXpertDataset(train_df, transform=train_transform, image_root=image_root)
    valid_dataset = CheXpertDataset(valid_df, transform=valid_transform, image_root=image_root)

    # Use the custom collate function
    train_loader = DataLoader(
        train_dataset,                   # ✅ Use dataset, not dataframe
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,                   # ✅ Use dataset, not dataframe
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
