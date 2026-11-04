import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GlossaryDisclosure } from "./GlossaryDisclosure";

describe("GlossaryDisclosure", () => {
  it("toggles its explanation from a real button with aria-expanded", () => {
    render(<GlossaryDisclosure text="Expected goals per chance." label="What is xG?" />);
    const button = screen.getByRole("button", { name: "What is xG?" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Expected goals per chance.")).toBeNull();

    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    const panel = document.getElementById(button.getAttribute("aria-controls") ?? "");
    expect(panel).toHaveTextContent("Expected goals per chance.");

    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Expected goals per chance.")).toBeNull();
  });
});
