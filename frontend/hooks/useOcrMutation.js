// hooks/useOcrMutation.js
import { useMutation } from '@tanstack/react-query';
import { extractTextFromFile } from '../utils/services/apis';

export const useOcrMutation = () => {
  return useMutation({
    mutationFn: ({ fileUri, fileType }) => extractTextFromFile(fileUri, fileType),
    onError: (error) => {
      console.error('OCR Mutation Error:', error);
      return error.message;
    }
  });
};