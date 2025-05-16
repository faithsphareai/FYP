import { useMutation } from '@tanstack/react-query';
import { generateQuiz, gradeQuiz } from '../utils/services/apis';

export const useQuizMutations = () => {
  const generateQuizMutation = useMutation({
    mutationFn: (search_query) => generateQuiz(search_query),
    onError: (error) => {
      console.error('Quiz Generation Mutation Error:', error);
      return error.message;
    }
  });

  const gradeQuizMutation = useMutation({
    mutationFn: (answers) => gradeQuiz(answers),
    onError: (error) => {
      console.error('Quiz Grading Mutation Error:', error);
      return error.message;
    }
  });

  return {
    generateQuizMutation,
    gradeQuizMutation
  };
};