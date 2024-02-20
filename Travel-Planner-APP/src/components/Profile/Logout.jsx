import React from "react";
import styled from "styled-components";
import { FaSignOutAlt as LogOut } from "react-icons/fa";
export default function Logout() {
  return (
    <Container>
      <LogOut className="icon" size="27" />
      <h2>Logout</h2>
    </Container>
  );
}
const Container = styled.div`
  cursor: pointer;
  width: 100%;
  display: flex;
  flex-direction: row;
  gap: 1rem;
  align-items: center;
  h2,
  .icon {
    color: white;
  }
`;
