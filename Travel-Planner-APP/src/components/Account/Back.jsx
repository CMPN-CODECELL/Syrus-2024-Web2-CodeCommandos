import React from "react";
import styled from "styled-components";
import { FaArrowLeft as Left } from "react-icons/fa";
export default function Back({ changeMenu, setMessages = null, setSearchWhat= null,inPosts}) {
  return (
    <Button
      onClick={() => {
        changeMenu("Profile");
        if(setMessages){
          setMessages(false)
        }
        if(setSearchWhat){
          setSearchWhat("Users")
        }
      }}
    >
      <Left size="20" />
    </Button>
  );
}
const Button = styled.button`
  border: none;
  color: white;
  background-color: transparent;
`;
