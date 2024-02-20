import React, { useState, useEffect, useCallback } from "react";
import styled from "styled-components";
import DatePicker from "react-datepicker";
import Select from "react-select";
import { getUsers, ticketUpload} from "../../utils/api-routes";
import "react-datepicker/dist/react-datepicker.css";
import axios from "axios";
import { useDropzone } from "react-dropzone";

export default function AddTrip({setAddShowTrip}) {
  const [departureDate, setDepartureDate] = useState(null);
  const [arrivalTime, setArrivalTime] = useState(null);
  const [returnDate, setReturnDate] = useState(null);
  const [arrivalCity, setArrivalCity] = useState("");
  const [tripMates, setTripMates] = useState([]);
  const [selectedMates, setSelectedMates] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const user = JSON.parse(localStorage.getItem("user"));

  useEffect(() => {
    async function friends() {
      try {
        console.log("Fetching friends");
        const { data } = await axios.get(getUsers, {
          headers: {
            Authorization: user.token,
          },
        });
        setTripMates(data);
      } catch (error) {
        console.log(error.message);
      }
    }
    friends();
  }, [user.token]);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    setSelectedFile(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: "image/*",
    multiple: false,
  });

  const handleCancelUpload = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("arrivalCity:", arrivalCity);
    const formattedDepartureDate =
      departureDate && departureDate.toLocaleDateString();
    const formattedReturnDate =
      returnDate && returnDate.toLocaleDateString();

    const formData = new FormData();
    formData.append("departureDate", formattedDepartureDate);
    formData.append(
      "arrivalTime",
      arrivalTime && arrivalTime.toLocaleTimeString()
    );
    formData.append("returnDate", formattedReturnDate);
    formData.append("arrivalCity", arrivalCity);

    formData.append(
      "tripMates",
      JSON.stringify(selectedMates.map((mate) => mate.value))
    );

    if (selectedFile) {
      formData.append("image", selectedFile);
    }

    try {
      const response = await axios.post(ticketUpload, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: user.token,
        },
      });
      console.log("Form submitted successfully:", response.data);
    } catch (error) {
      console.error("Error submitting form:", error.message);
    }
  };

  const allTripMates = tripMates.map((item) => ({
    value: item._id,
    label: item.name,
  }));

  return (
    <Container>
      <form onSubmit={(e) => handleSubmit(e)}>
        <FormGroup>
          <label>Departure Date:</label>
          <DatePicker
            selected={departureDate}
            onChange={(date) => setDepartureDate(date)}
            dateFormat="MMMM d, yyyy"
            minDate={new Date()} // Set the minimum date to today
          />
        </FormGroup>

        <FormGroup>
          <label>Return Date:</label>
          <DatePicker
            selected={returnDate}
            onChange={(date) => setReturnDate(date)}
            dateFormat="MMMM d, yyyy"
            minDate={departureDate} // Set the minimum date to the selected departure date
          />
        </FormGroup>

        <FormGroup>
          <label>Departure Time:</label>
          <DatePicker
            selected={arrivalTime}
            onChange={(time) => setArrivalTime(time)}
            showTimeSelect
            showTimeSelectOnly
            timeIntervals={15}
            dateFormat="h:mm aa"
          />
        </FormGroup>

        <FormGroup>
          <label>Arrival City:</label>
          <input
            type="text"
            value={arrivalCity}
            onChange={(e) => setArrivalCity(e.target.value)}
            placeholder="Enter City..."
          />
        </FormGroup>

        <FormGroup>
          <label>Add Trip Mates:</label>
          <Select
            isMulti
            options={allTripMates}
            value={selectedMates}
            onChange={(selectedOptions) => {
              setSelectedMates(selectedOptions);
            }}
            placeholder="Select Trip Mates..."
          />
        </FormGroup>

        <DropzoneContainer
          {...getRootProps()}
          name="profilePicture"
          isDragActive={isDragActive}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the files here ...</p>
          ) : (
            <p>Drag 'n' drop a profile picture, or click to select one</p>
          )}
          {selectedFile && (
            <div className="image-preview">
              <p>Selected Image Preview</p>
              <img src={URL.createObjectURL(selectedFile)} alt="dp" />
              <button onClick={handleCancelUpload}>Cancel Upload</button>
            </div>
          )}
        </DropzoneContainer>
        <SubmitButton type="submit">Submit</SubmitButton>
      </form>
      <FloatingActionButton onClick={()=>setAddShowTrip("show")}>X</FloatingActionButton>
    </Container>
  );
}

const Container = styled.div`
  height: 100%;
  width: 100%;
  background-color: #282a34;
  border-radius: 4rem;
  padding: 20px;
`;

const FormGroup = styled.div`
  margin-bottom: 20px;
  label {
    display: block;
    color: white;
    margin-bottom: 5px;
  }

  input {
    width: 100%;
    padding: 10px;
    border: 1px solid #61dafb;
    border-radius: 5px;
    color: white;
    background-color: #282a34;
    outline: none;
  }
`;

const DropzoneContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 2px dashed #61dafb;
  border-radius: 4px;
  background-color: ${(props) => (props.isDragActive ? "#4e0eff" : "#282a34")};
  padding: 20px;
  cursor: pointer;
  margin-bottom: 20px;
  p {
    color: white;
  }
  .image-preview {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    p {
      color: white;
      align-self: center;
    }
    img {
      width: 100px;
      height: 100px;
      margin-top: 5px;
      align-self: center;
    }
    button {
      background-color: #997af0;
      color: white;
      width: fit-content;
      padding: 0.3rem 0.3rem;
      border: none;
      font-weight: bold;
      cursor: pointer;
      border-radius: 0.4rem;
      font-size: 1rem;
      text-transform: uppercase;
      transition: 0.5s ease-in-out;
      &:hover {
        background-color: #4e0eff;
      }
    }
  }
`;

const SubmitButton = styled.button`
  background-color: #61dafb;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
`;

const FloatingActionButton = styled.button`
  position: fixed;
  bottom: 20px;
  right: 20px;
  background-color: #61dafb;
  color: white;
  padding: 15px;
  border: none;
  font-size: 1.5em;
  border-radius: 50%;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease-in-out;

  &:hover {
    background-color: #4e0eff;
  }
`;