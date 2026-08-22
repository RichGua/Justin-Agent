import type { ClientContext } from '@deepseek-ai/dsh-client-runtime/client'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type {} from '@deepseek-ai/dsh-client-ui-sidebar/client'

function RundeepMark({ size, className }: { size?: number | undefined; className?: string | undefined }) {
  const dimension = size ?? 28
  return (
    <svg className={className} width={dimension} height={dimension} viewBox="0 0 64 64" aria-hidden="true">
      <path d="M15 50V14h19c9 0 15 5.1 15 12.2S43 38.4 34 38.4H15" fill="none" stroke="currentColor" strokeWidth="6.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m30.6 38.4 17.5 11.7" fill="none" stroke="currentColor" strokeWidth="6.4" strokeLinecap="round" />
      <circle cx="15" cy="14" r="1.65" fill="currentColor" />
    </svg>
  )
}

function RundeepName() {
  return <span className="rundeepBrandName">Rundeep</span>
}

/** Replace upstream-owned visual slots without changing the DSH runtime packages. */
export function applyRundeepBrand(ctx: ClientContext): void {
  ctx.slots.inject('sidebar.brand.mark', () => ctx.slots.inject('sidebar.brand.name', () => ctx.slots.inject('conversation.hero.brand.mark', function* () {
    yield ctx.slots.register({ name: 'sidebar.brand.mark' }, RundeepMark)
    yield ctx.slots.register({ name: 'sidebar.brand.name' }, RundeepName)
    yield ctx.slots.register({ name: 'conversation.hero.brand.mark' }, RundeepMark)
  })))
}
