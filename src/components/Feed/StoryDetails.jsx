import React from "react";
import styled from "styled-components";
import Logo from "../../assets/download.jpeg";
import DM from "../Account/DM";
export default function StoryDetails() {
  const arr = [1, 2, 3, 1, 1, 9, 7, 6];
  return (
    <Container>
      <div className="story-details">
        {arr.map(() => (
          <div className="image-container">
            <img src={Logo} alt="image" />
            <DM/>
          </div>
        ))}
      </div>
    </Container>
  );
}
const Container = styled.div`
  user-select: none;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100vh;
  overflow: hidden;
  .story-details {
    margin: 1rem;
    padding: 1rem;
    display: flex;
    flex-direction: row;
    gap: 0.1rem;
    height: 95%;
    width: 95%;
    background-color: #282a34;
    border-radius: 1rem;
    overflow-x: auto;
    &::-webkit-scrollbar {
      cursor: pointer;
      padding: 1rem;
      height: 0.4rem;
      width: 0.2rem;
      &-thumb {
        background-color: #ffffff39;
        width: 0.1rem;
        border-radius: 1rem;
      }
    }
    .image-container {
      flex-shrink: 0;
      height: 99%;
      width: 50%;
      border-radius: 0.5rem;
      border: 0.1rem solid #4c3eff;
      background-color: #3c3f51;
      img {
        padding: 0.1rem;
        height: 90%;
        width: 100%;
        border-radius: 0.5rem;
      }
    }
  }
`;
