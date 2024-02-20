import React from "react";
import styled from "styled-components";
import Logo from "../../assets/logo.png";
export default function ProfileLogo() {
  return (
    <Container>
      <img src={Logo} alt="logo" />
      <h2>
        <span className="e">T</span>
        <span className="c">R</span>
        <span className="h">A</span>
        <span className="o">V</span>
        <span className="s">E</span>
        <span className="e">L</span>
      </h2>
    </Container>
  );
}
const Container = styled.div`
user-select: none;
  width: 100%;
  gap: 3rem;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  img {
    height: 3rem;
  }
  h2 {
    span{
      text-decoration: underline;
    }
    .e{
      color: yellow;
    }
    .c{
      color: red;
    }
    .h{
      color: green;
    }
    .o{
      color: blue;
    }
    .s{
      color: orange;
    }
  }
`;
