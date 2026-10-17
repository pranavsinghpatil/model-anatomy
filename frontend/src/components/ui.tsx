// Radix composition patterns adapted from shadcn/ui (MIT); see THIRD_PARTY.md.
import * as Dropdown from "@radix-ui/react-dropdown-menu";
import * as Dialog from "@radix-ui/react-dialog";
import { cva, type VariantProps } from "class-variance-authority";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { Check, Monitor, Moon, Sun, X } from "lucide-react";
import { type ButtonHTMLAttributes, type ReactNode } from "react";
import { ButtonCopy } from "./vendor/button-copy";
import { usePreferences } from "../state/preferences";
export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));
const variants = cva("button", {
  variants: {
    variant: {
      primary: "button-primary",
      secondary: "button-secondary",
      ghost: "button-ghost",
    },
  },
  defaultVariants: { variant: "secondary" },
});
export function Button({
  className,
  variant,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof variants>) {
  return <button className={cn(variants({ variant }), className)} {...props} />;
}
export function Menu({
  label,
  children,
  trigger,
  disabled = false,
}: {
  label: string;
  children: ReactNode;
  trigger: ReactNode;
  disabled?: boolean;
}) {
  return (
    <Dropdown.Root>
      <Dropdown.Trigger asChild>
        <Button aria-label={label} disabled={disabled}>
          {trigger}
        </Button>
      </Dropdown.Trigger>
      <Dropdown.Portal>
        <Dropdown.Content
          className="menu-content"
          sideOffset={8}
          align="end"
          collisionPadding={12}
        >
          {children}
        </Dropdown.Content>
      </Dropdown.Portal>
    </Dropdown.Root>
  );
}
export function MenuItem({
  children,
  onSelect,
  disabled,
}: {
  children: ReactNode;
  onSelect: () => void;
  disabled?: boolean;
}) {
  return (
    <Dropdown.Item
      className="menu-item"
      onSelect={onSelect}
      disabled={disabled}
    >
      {children}
    </Dropdown.Item>
  );
}
export function ThemeControl() {
  const { theme, setTheme } = usePreferences();
  return (
    <Menu
      label="Theme"
      trigger={
        <>
          <span className="theme-glyph" aria-hidden="true" data-theme={theme}>
            <Sun size={15} className="theme-light" />
            <Moon size={15} className="theme-dark" />
            <Monitor size={15} className="theme-system" />
          </span>
          <span className="wide-label">Theme</span>
        </>
      }
    >
      <Dropdown.RadioGroup
        value={theme}
        onValueChange={(value) => setTheme(value as typeof theme)}
      >
        {(
          [
            ["light", Sun],
            ["dark", Moon],
            ["system", Monitor],
          ] as const
        ).map(([value, Icon]) => (
          <Dropdown.RadioItem key={value} value={value} className="menu-item">
            <Icon size={16} />
            {value[0].toUpperCase() + value.slice(1)}
            <Dropdown.ItemIndicator>
              <Check size={14} />
            </Dropdown.ItemIndicator>
          </Dropdown.RadioItem>
        ))}
      </Dropdown.RadioGroup>
    </Menu>
  );
}
export function Drawer({
  open,
  onOpenChange,
  title,
  trigger,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  trigger: ReactNode;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Trigger asChild>{trigger}</Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="drawer">
          <div className="panel-heading">
            <Dialog.Title>{title}</Dialog.Title>
            <Dialog.Close asChild>
              <Button
                variant="ghost"
                aria-label={`Close ${title.toLowerCase()}`}
              >
                <X size={18} />
              </Button>
            </Dialog.Close>
          </div>
          <Dialog.Description className="sr-only">
            Select a component in the current graph. Open its nested graph with
            its separate action.
          </Dialog.Description>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
export function CopyButton({
  value,
  label = "Copy",
}: {
  value: string;
  label?: string;
}) {
  return <ButtonCopy key={value} value={value} label={label} />;
}
