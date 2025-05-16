import React, { useState, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Animated,
  Dimensions,
  SafeAreaView,
} from "react-native";
import { useRouter } from "expo-router";

const questions = [
  {
    question: "How are you feeling physically today?",
    options: ["Energetic", "Tired", "Sick", "Neutral"],
  },
  {
    question: "How would you describe your mood right now?",
    options: ["Happy", "Sad", "Excited", "Indifferent"],
  },
  {
    question: "Are you feeling stressed or relaxed today?",
    options: ["Very Stressed", "Somewhat Stressed", "Relaxed", "Very Relaxed"],
  },
  {
    question: "Have you felt overwhelmed or anxious recently?",
    options: ["Yes, often", "Sometimes", "Rarely", "Not at all"],
  },
];

const { width } = Dimensions.get("window");

const MoodQuestionnaire = () => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(-1);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const router = useRouter();
  const slideAnim = useRef(new Animated.Value(0)).current;
  const [direction, setDirection] = useState("right"); // Track animation direction

  const handleStart = () => {
    setCurrentQuestionIndex(0);
    slideAnim.setValue(0);
  };

  const animateToNextQuestion = (nextIndex, animDirection) => {
    setDirection(animDirection);
    const startValue = animDirection === "right" ? width : -width;
    const endValue = 0;

    slideAnim.setValue(startValue);
    setCurrentQuestionIndex(nextIndex);

    Animated.spring(slideAnim, {
      toValue: endValue,
      useNativeDriver: true,
      friction: 8,
      tension: 40,
    }).start();
  };

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      if (selectedAnswers[currentQuestionIndex] !== undefined) {
        const nextIndex = currentQuestionIndex + 1;
        animateToNextQuestion(nextIndex, "right");
      }
    } else {
        router.push({
            pathname: "/chatbot",
            params: {
              chatContext: "waqiya",
              title: "Waqiya Recommendation",
            },
          });
        }
      };

  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      const prevIndex = currentQuestionIndex - 1;
      animateToNextQuestion(prevIndex, "left");
    }
  };

  const handleOptionSelect = (index, option) => {
    setSelectedAnswers((prev) => ({ ...prev, [index]: option }));
  };

  const currentQuestion = questions[currentQuestionIndex];

  return (
    <SafeAreaView className="flex-1 bg-white">
      {currentQuestionIndex === -1 ? (
        // Welcome Screen remains the same
        <View className="flex-1 justify-center items-center px-6">
          <Text className="text-3xl font-bold text-center mb-6">
            Daily Reflection
          </Text>
          <Text className="text-lg text-center mb-10 text-gray-700">
            Take a moment to check in with yourself. Answer these questions to
            gain insights into your emotional and spiritual well-being.
          </Text>
          <TouchableOpacity
            className="bg-[#0b8c5c] px-8 py-4 rounded-xl shadow-md"
            onPress={handleStart}
          >
            <Text className="text-white text-lg font-bold">
              Begin Reflection
            </Text>
          </TouchableOpacity>
        </View>
      ) : (
        // Question Screen with Animation
        <View className="flex-1 justify-center items-center px-6">
          <Text className="absolute top-4 right-4 text-gray-500">
            {currentQuestionIndex + 1} / {questions.length}
          </Text>

          <Animated.View
            style={{
              width: "100%",
              transform: [{ translateX: slideAnim }],
            }}
          >
            <Text className="text-2xl font-bold text-center mb-10 ">
              {currentQuestion.question}
            </Text>

            <View className="w-full">
              {currentQuestion.options.map((option, optionIndex) => (
                <TouchableOpacity
                  key={optionIndex}
                  className={`px-5 py-4 rounded-lg mb-4 ${
                    selectedAnswers[currentQuestionIndex] === option
                      ? "bg-[#0b8c5c]"
                      : "bg-gray-200"
                  }`}
                  onPress={() =>
                    handleOptionSelect(currentQuestionIndex, option)
                  }
                >
                  <Text
                    className={`text-lg text-center ${
                      selectedAnswers[currentQuestionIndex] === option
                        ? "text-white"
                        : "text-gray-800"
                    }`}
                  >
                    {option}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </Animated.View>

          <View className="flex-row justify-between mt-8 w-full px-4">
            {currentQuestionIndex > 0 && (
              <TouchableOpacity
                className="bg-gray-200 px-6 py-3 rounded-lg"
                onPress={handleBack}
              >
                <Text className="text-gray-800 text-lg">Back</Text>
              </TouchableOpacity>
            )}

            <TouchableOpacity
              className={`px-6 py-3 rounded-lg ${
                selectedAnswers[currentQuestionIndex] !== undefined
                  ? "bg-[#0b8c5c]"
                  : "bg-gray-300"
              }`}
              onPress={handleNext}
              disabled={selectedAnswers[currentQuestionIndex] === undefined}
            >
              <Text
                className={`text-lg ${
                  selectedAnswers[currentQuestionIndex] !== undefined
                    ? "text-white"
                    : "text-gray-500"
                }`}
              >
                {currentQuestionIndex === questions.length - 1
                  ? "Finish"
                  : "Next"}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
};

export default MoodQuestionnaire;
