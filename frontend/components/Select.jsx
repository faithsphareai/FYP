import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Modal, Animated } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

export function Select({ value, onValueChange, children, className }) {
  const [isOpen, setIsOpen] = useState(false);
  const slideAnim = new Animated.Value(0);

  const toggleSelect = () => {
    setIsOpen(!isOpen);
    Animated.spring(slideAnim, {
      toValue: isOpen ? 0 : 1,
      useNativeDriver: true,
    }).start();
  };

  return (
    <>
      <TouchableOpacity 
        onPress={toggleSelect}
        className={`flex-row items-center justify-between p-3 bg-white ${className}`}
      >
        <Text className="text-green-800">{value}</Text>
        <Ionicons name={isOpen ? "chevron-up" : "chevron-down"} size={20} color="green" />
      </TouchableOpacity>

      <Modal
        visible={isOpen}
        transparent
        animationType="fade"
        onRequestClose={() => setIsOpen(false)}
      >
        <TouchableOpacity 
          className="flex-1 bg-black/20"
          onPress={() => setIsOpen(false)}
        >
          <Animated.View 
            style={{
              transform: [{
                translateY: slideAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [100, 0]
                })
              }]
            }}
            className="mt-auto bg-white rounded-t-xl"
          >
            {React.Children.map(children, child =>
              React.cloneElement(child, {
                onSelect: (value) => {
                  onValueChange(value);
                  setIsOpen(false);
                }
              })
            )}
          </Animated.View>
        </TouchableOpacity>
      </Modal>
    </>
  );
}

Select.Item = function SelectItem({ label, value, onSelect }) {
  return (
    <TouchableOpacity
      onPress={() => onSelect(value)}
      className="p-4 border-b border-gray-100"
    >
      <Text className="text-green-800">{label}</Text>
    </TouchableOpacity>
  );
};