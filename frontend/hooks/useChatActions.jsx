import { useChat } from "../context/ChatContext";
import { formatMessage } from "../utils/messageFormat";
import { saveChatHistory } from "../utils/chatStorage";
import { useAyatFinderMutation } from "./useAyatFinderMutation";
import { useHadithFinderMutation } from "./useHadithFinderMutation";
import { useQuizMutations } from "./useQuizMutations";
import { useGeneralChatbotMutation } from "./useGeneralChatbot";

export const useChatActions = () => {
  const { messages, setMessages, chatContext } = useChat();
  const ayatFinderMutation = useAyatFinderMutation();
  const hadithFinderMutation = useHadithFinderMutation();
  const generalChatbotMutation = useGeneralChatbotMutation();
  const { generateQuizMutation } = useQuizMutations();

  const startNewQuiz = async () => {
    const typingMessage = formatMessage("Typing", "bot", {
      isTyping: true,
      animatedText: true,
      dots: "...",
    });
    setMessages([typingMessage]);

    try {
      const response = await generateQuizMutation.mutateAsync("new topic");
      let botMessage = formatMessage(response, "bot", { isQuestion: true });
      setMessages([botMessage]);
      await saveChatHistory(chatContext, [botMessage]);
    } catch (error) {
      const errorMessage = formatMessage(
        "Sorry, I encountered an error. Please try again.",
        "bot"
      );
      setMessages([errorMessage]);
    }
  };

  const sendMessage = async (text) => {
    const userMessage = formatMessage(text, "user");
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    const typingMessage = formatMessage("Typing", "bot", {
      isTyping: true,
      animatedText: true,
      dots: "...",
    });
    setMessages((prevMessages) => [...prevMessages, typingMessage]);

    let dotsCount = 0;
    const typingInterval = setInterval(() => {
      dotsCount = (dotsCount + 1) % 4;
      const dots = ".".repeat(dotsCount);
      setMessages((prevMessages) =>
        prevMessages.map((msg) =>
          msg.isTyping ? { ...msg, text: `Typing${dots}` } : msg
        )
      );
    }, 500);

    try {
      let response;

      if (chatContext === "quiz") {
        // For quiz context, first generate a question, then grade the answer
        const lastBotMessage = messages.filter((m) => m.sender === "bot").pop();
        if (lastBotMessage?.isQuestion) {
          // Generate next question after grading
          const nextQuestion = await generateQuizMutation.mutateAsync(text);
          response = `${nextQuestion}`;
        } else {
          // Generate first question
          response = await generateQuizMutation.mutateAsync(text);
        }
      } else if (chatContext === "quran") {
        // Use Ayat Finder for Quran authentication
        response = await ayatFinderMutation.mutateAsync(text);
      } else if (chatContext === "hadith") {
        // Use Hadith Finder for Hadith authentication
        response = await hadithFinderMutation.mutateAsync(text);
      } else if (chatContext === "generalChatbot") {
        // Use General Chatbot for general queries
        response = await generalChatbotMutation.mutateAsync(text);
      } else {
        throw new Error(`Unsupported chat context: ${chatContext}`);
      }

      clearInterval(typingInterval);

      if (!response) {
        throw new Error("No response received from the API");
      }

      const botMessage = formatMessage(response, "bot", {
        isQuestion: chatContext === "quiz",
      });

      setMessages((prevMessages) =>
        prevMessages.filter((msg) => !msg.isTyping).concat(botMessage)
      );

      await saveChatHistory(chatContext, [
        ...messages,
        userMessage,
        botMessage,
      ]);
    } catch (error) {
      console.error("Chat action error:", error);
      clearInterval(typingInterval);

      const errorMessage = formatMessage(
        `Sorry, I encountered an error: ${error.message}. Please try again.`,
        "bot"
      );
      setMessages((prevMessages) =>
        prevMessages.filter((msg) => !msg.isTyping).concat(errorMessage)
      );
    }
  };

  return { sendMessage, startNewQuiz };
};

// hooks/useChatActions.js

// import { useChat } from "../context/ChatContext";
// import { formatMessage } from "../utils/messageFormat";
// import { saveChatHistory } from "../utils/chatStorage";
// import { useAyatFinderMutation } from "./useAyatFinderMutation";
// import { useHadithFinderMutation } from "./useHadithFinderMutation";
// import { useQuizMutations } from "./useQuizMutations";
// import { useGeneralChatbotMutation } from "./useGeneralChatbot";

// export const useChatActions = () => {
//   const { messages, setMessages, chatContext } = useChat();
//   const ayatFinderMutation = useAyatFinderMutation();
//   const hadithFinderMutation = useHadithFinderMutation();
//   const generalChatbotMutation = useGeneralChatbotMutation();
//   const { generateQuizMutation, gradeQuizMutation } = useQuizMutations();

//   const startNewQuiz = async () => {
//     const typing = formatMessage("Typing", "bot", {
//       isTyping: true,
//       animatedText: true,
//       dots: "...",
//     });
//     setMessages([typing]);

//     try {
//       const questionText = await generateQuizMutation.mutateAsync("new topic");
//       const botMessage = formatMessage(questionText, "bot", { isQuestion: true });
//       setMessages([botMessage]);
//       await saveChatHistory(chatContext, [botMessage]);
//     } catch (error) {
//       const errorMessage = formatMessage(
//         "Sorry, I encountered an error. Please try again.",
//         "bot"
//       );
//       setMessages([errorMessage]);
//     }
//   };

//   const sendMessage = async (text) => {
//     // 1) Add user message
//     const userMessage = formatMessage(text, "user");
//     setMessages(prev => [...prev, userMessage]);

//     // 2) Show typing indicator
//     const typing = formatMessage("Typing", "bot", {
//       isTyping: true,
//       animatedText: true,
//       dots: "...",
//     });
//     setMessages(prev => [...prev, typing]);

//     // Animated dots
//     let dotsCount = 0;
//     const typingInterval = setInterval(() => {
//       dotsCount = (dotsCount + 1) % 4;
//       const dots = ".".repeat(dotsCount);
//       setMessages(prev =>
//         prev.map(m => m.isTyping ? { ...m, text: `Typing${dots}` } : m)
//       );
//     }, 500);

//     try {
//       let botMessage;        // <-- only one declaration
//       let apiResponse;

//       if (chatContext === "quiz") {
//         // get last bot msg to see if it was a question
//         const lastBot = [...messages].reverse().find(m => m.sender === "bot");

//         if (lastBot?.isQuestion) {
//           // Grade the answer
//           apiResponse = await gradeQuizMutation.mutateAsync(text);
//           botMessage = formatMessage(apiResponse, "bot", {
//             isQuestion: false
//           });
//         } else {
//           // Generate a new quiz question
//           apiResponse = await generateQuizMutation.mutateAsync(text);
//           botMessage = formatMessage(apiResponse, "bot", {
//             isQuestion: true
//           });
//         }

//       } else if (chatContext === "quran") {
//         apiResponse = await ayatFinderMutation.mutateAsync(text);
//         botMessage = formatMessage(apiResponse, "bot");
//       } else if (chatContext === "hadith") {
//         apiResponse = await hadithFinderMutation.mutateAsync(text);
//         botMessage = formatMessage(apiResponse, "bot");
//       } else if (chatContext === "generalChatbot") {
//         apiResponse = await generalChatbotMutation.mutateAsync(text);
//         botMessage = formatMessage(apiResponse, "bot");
//       } else {
//         throw new Error(`Unsupported chat context: ${chatContext}`);
//       }

//       clearInterval(typingInterval);

//       // Replace typing with the actual bot message
//       setMessages(prev =>
//         prev
//           .filter(m => !m.isTyping)
//           .concat(botMessage)
//       );

//       // Persist
//       await saveChatHistory(chatContext, [...messages, userMessage, botMessage]);

//     } catch (error) {
//       clearInterval(typingInterval);
//       console.error("Chat action error:", error);
//       const errorMessage = formatMessage(
//         `Sorry, I encountered an error: ${error.message}. Please try again.`,
//         "bot"
//       );
//       setMessages(prev =>
//         prev.filter(m => !m.isTyping).concat(errorMessage)
//       );
//     }
//   };

//   return { sendMessage, startNewQuiz };
//};
