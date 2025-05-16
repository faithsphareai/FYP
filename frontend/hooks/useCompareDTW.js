import { useMutation } from '@tanstack/react-query';
import { compareDTW } from '../utils/services/apis';

export const useCompareDTW = () => {
  return useMutation({
    mutationFn: ({ originalAudio, userAudio }) => compareDTW(originalAudio, userAudio),
    onError: (error) => {
      console.error('DTW Comparison Mutation Error:', error);
      return error.message;
    }
  });
};