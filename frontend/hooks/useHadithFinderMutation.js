import { useMutation } from '@tanstack/react-query';
import { getHadithAnswer } from '../utils/services/apis';
import {useChat} from "../context/ChatContext"

export const useHadithFinderMutation = () => {
  const {language} = useChat();
  return useMutation({
    mutationFn: (question) => getHadithAnswer(question,language),
    onError: (error) => {
      console.error('Hadith Finder Mutation Error:', error);
      return error.message;
    }
  });
};