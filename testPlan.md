# Voice Agent - Test plan end-to-end (Ladder + Compliance)

---

## 1. Flujo ideal (pago completo inmediato)

| Type | Expected text |
|------|--------------|
| Usuario | Hi, I want to pay my overdue balance. |
| Agente | Great, I can help you with that today. Your past due balance is $X. Would you like to pay the full amount today? |
| Usuario | Yes, I’ll pay it in full. |
| Agente | Perfect. I’ve processed your full payment. Your account will be updated shortly. |

---

## 2. Rechaza nivel 1 → acepta nivel 2

| Type | Expected text |
|------|--------------|
| Usuario | I can’t pay that much. |
| Agente | The best option is paying the full past due balance today. |
| Usuario | That’s too much. |
| Agente | I understand. We can offer a reduced payment plan. Would you like to proceed with that option? |
| Usuario | Yes. |
| Agente | Great, I’ve set up a reduced payment arrangement for you. |

---

## 3. Rechaza nivel 1 y 2 → acepta plan de pagos (nivel 3)

| Type | Expected text |
|------|--------------|
| Usuario | I still can’t pay that. |
| Agente | Let’s explore a structured payment plan over time. |
| Usuario | Okay, I can do installments. |
| Agente | Perfect. I’ve arranged a payment plan with scheduled installments. |

---

## 4. Rechaza todo → partial settlement (nivel 4)

| Type | Expected text |
|------|--------------|
| Usuario | I can’t commit to any of those options. |
| Agente | I may be able to offer a partial settlement option. |
| Usuario | I can only pay part of it. |
| Agente | Understood. I’ve recorded a partial settlement option for your account. |

---

## 5. Solicitud de agente humano (en cualquier punto)

| Type | Expected text |
|------|--------------|
| Usuario | I want to talk to a human agent. |
| Agente | I understand. I will transfer you to a live representative. |

---

## 6. Fallo de verificación de identidad

| Type | Expected text |
|------|--------------|
| Usuario | My name is John Doe. |
| Agente | I’m unable to verify your identity with the information provided. Please try again or contact support. |

---

## 7. Hardship (pérdida de empleo / enfermedad)

| Type | Expected text |
|------|--------------|
| Usuario | I lost my job recently. |
| Agente | I’m really sorry to hear that. Let’s look for options that can help you during this time. |

---

## 8. Pregunta sobre crédito

| Type | Expected text |
|------|--------------|
| Usuario | Will this affect my credit score? |
| Agente | Depending on your account status, missed payments may be reported to credit bureaus. |

---

## 9. Solicitud de no contacto

| Type | Expected text |
|------|--------------|
| Usuario | Stop calling me. |
| Agente | I will update your account preferences to reduce or stop communications as requested. |

---

## 10. Usuario agresivo o termina llamada

| Type | Expected text |
|------|--------------|
| Usuario | This is harassment, stop calling me! |
| Agente | I understand your frustration. I will close this interaction now. |

---

## 11. Disputa de deuda

| Type | Expected text |
|------|--------------|
| Usuario | I already paid this. |
| Agente | I will mark this account as disputed and escalate it for review. |

---

## 12. Restricción de contacto (horarios / trabajo)

| Type | Expected text |
|------|--------------|
| Usuario | Don’t call me at work. |
| Agente | I will update your contact preferences and respect your communication restrictions. |

---

## 13. Llamada caída (recuperación de estado)

| Type | Expected text |
|------|--------------|
| Sistema | WebRTC disconnected |
| Agente (reconnect) | Welcome back. We can continue where we left off regarding your account options. |

---