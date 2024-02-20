import React, { useState } from "react";
import styled from "styled-components";
import Feed from "../components/Feed";
import Account from "../components/Account";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
export default function Home() {
  const navigate = useNavigate();
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem('user'))
    console.log(!user);
      if(!user){
        navigate('/auth')
    };
  },[navigate]);
  const [messages, setMessages] = useState(false);
  const [menu, setMenu] = useState("Profile");
  const [feed,setFeed] = useState("home")
  const changeMenu = (menu) => {
    setMenu(menu);
  };
  return (
    <Container messages={messages} style={{gridTemplateColumns: messages ? '30% 70%': '20% 80%'}}>
      <Account setMessages={setMessages} changeMenu={changeMenu} menu={menu} setFeed={setFeed}/>
      <Feed feed={feed} setFeed={setFeed}/>
    </Container>
  );
}
const Container = styled.div`
  height: 100vh;
  width: 100vw;
  display: grid;
  grid-template-columns: ${(props) => (props.messages ? "30% 70%" : "20% 80%")};
  justify-content: center;
  align-items: center;
  background-color: #282a34;
  transition: grid-template-columns 0.5s ease-in-out;
`;
