import { useMutation } from '@tanstack/react-query';
import { getAyatAnswer } from '../utils/services/apis';
import {useChat} from "../context/ChatContext"

export const useAyatFinderMutation = () => {

  const {language} = useChat();

  return useMutation({
    mutationFn: (question) => getAyatAnswer(question, language),
    onError: (error) => {
      console.error('Ayat Finder Mutation Error:', error);
      return error.message;
    }
  });
};