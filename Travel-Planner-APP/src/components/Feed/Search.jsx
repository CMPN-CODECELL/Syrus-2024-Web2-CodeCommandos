import React from "react";
import styled from "styled-components";
import Hotel from "./Hotel";
import Flight from "./Flight";
import Activity from "./Activity";
export default function Search() {
  return <Container>
    <Hotel/>
    <Flight/>
  </Container>;
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  display: flex;
`;
