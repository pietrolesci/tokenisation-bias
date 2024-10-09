from datatrove.data import DocumentsPipeline
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer, TokenizedFile
from datatrove.utils.batching import batched


class TokenizedDocumentDataset(DocumentTokenizer):
    def write_unshuffled(self, data: DocumentsPipeline, filename: str) -> TokenizedFile:  # type: ignore
        unshuff = TokenizedFile(
            self.output_folder if not self.shuffle or not self.local_working_dir else self.local_working_dir,
            filename,
            save_index=not self.shuffle,
            save_loss_metadata=self.save_loss_metadata,
            upload_block_size=self.upload_block_size,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            save_final_metadata=self.save_final_metadata,
            token_size=self.token_size,
        )
        # tokenize document's text in batches to go faster – we compute loss values independently if needed
        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                encoded_batch: list[list[int]] = [document.text for document in batch]
                for tokens in encoded_batch:
                    # write bytes to disk
                    unshuff.write(tokens, None)
                    # save stats
                    self.stat_update("tokens", value=len(tokens))
        unshuff.close()
        return unshuff
