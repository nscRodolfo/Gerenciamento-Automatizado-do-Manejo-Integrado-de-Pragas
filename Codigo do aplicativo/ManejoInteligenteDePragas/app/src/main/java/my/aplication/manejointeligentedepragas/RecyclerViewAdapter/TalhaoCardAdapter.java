package my.aplication.manejointeligentedepragas.RecyclerViewAdapter;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.AcoesCultura;
import my.aplication.manejointeligentedepragas.AdicionarTalhao;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.R;
import my.aplication.manejointeligentedepragas.Talhoes;
import my.aplication.manejointeligentedepragas.model.TalhaoModel;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class TalhaoCardAdapter extends RecyclerView.Adapter<TalhaoCardAdapter.ViewHolder>{

    private static final String TAG = "PropriedadeCardAdapter";

    private ArrayList<TalhaoModel> cards = new ArrayList<>();
    private Context talhaoContext;
    private String nomeTalhao;
    private int cod_Cultura;
    private int cod_Propriedade;
    private String nomePropriedade;
    private String nomeCultura;
    private  int Cod_Planta;

    String diasPraContagem;

    public TalhaoCardAdapter(Context talhaoContext, ArrayList<TalhaoModel> cards, int cod_Cultura, int cod_Propriedade, String nomePropriedade, String nomeCultura, int cod_Planta) {
        this.cards = cards;
        this.talhaoContext = talhaoContext;
        this.cod_Cultura = cod_Cultura;
        this.cod_Propriedade = cod_Propriedade;
        this.nomePropriedade= nomePropriedade;
        this.nomeCultura = nomeCultura;
        this.Cod_Planta=cod_Planta;
    }

    @NonNull
    @Override
    // resposável por inflar a view

    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.card_talhao, parent, false);
        ViewHolder holder = new ViewHolder(view);
        return holder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, final int position) {
        Log.d(TAG, "onBindViewHolder: called.");
        holder.talhao.setText(cards.get(position).getNome());
        Controller_Usuario cu = new Controller_Usuario(talhaoContext);
        holder.tipoUsu = cu.getUser().getTipo();

        CalculaDiasPraContagem(cards.get(position).getCod_Talhao(), holder, position);


        // true para, false continua pra proxima tela. função para click demorado

        if(holder.tipoUsu.equals("Produtor")) {
            holder.parent_layout_talhao.setOnLongClickListener(new View.OnLongClickListener() {
                @Override
                public boolean onLongClick(View v) {
                    exibirCaixaDialogo(cards.get(position));
                    return true;
                }
            });
        }
        holder.parent_layout_talhao.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "onClick:  clicked on: ");
                Intent i = new Intent(talhaoContext, AcoesCultura.class);
                nomeTalhao = cards.get(position).getNome();
                // chama a intent nesse adapter
                // pega o contexr do construtor
                int codigo = cards.get(position).getCod_Talhao();
                boolean aplicado = cards.get(position).isAplicado();
                i.putExtra("Cod_Talhao", codigo);
                i.putExtra("NomeTalhao", nomeTalhao);
                i.putExtra("Cod_Propriedade", cod_Propriedade);
                i.putExtra("nomePropriedade", nomePropriedade);
                i.putExtra("Cod_Cultura", cod_Cultura);
                i.putExtra("NomeCultura", nomeCultura);
                i.putExtra("Aplicado", aplicado);
                i.putExtra("Cod_Planta", Cod_Planta);
                talhaoContext.startActivity(i);
            }
        });


    }

    @Override
    public int getItemCount() {
        return cards.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        TextView talhao;
        TextView contagem;

        RelativeLayout parent_layout_talhao;


        String tipoUsu;


        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            talhao = itemView.findViewById(R.id.tvTalhao);
            contagem = itemView.findViewById(R.id.tvDiasParaContagemTalhao);
            parent_layout_talhao = itemView.findViewById(R.id.parent_layout_talhao);

        }
    }

    public void exibirCaixaDialogo(final TalhaoModel p)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(talhaoContext);
        dlgBox.setTitle("Alterar talhao");
        dlgBox.setMessage("Deseja excluir ou editar o talhao?");
        dlgBox.setPositiveButton("Editar", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                Intent k = new Intent(talhaoContext, AdicionarTalhao.class);
                k.putExtra("Cod_Propriedade", cod_Propriedade);
                k.putExtra("nomePropriedade", nomePropriedade);
                k.putExtra("Cod_Cultura", cod_Cultura);
                k.putExtra("NomeCultura", nomeCultura);
                k.putExtra("Aplicado", p.isAplicado());
                k.putExtra("NomeTalhao", p.getNome());
                k.putExtra("Cod_Talhao", p.getCod_Talhao());
                talhaoContext.startActivity(k);
            }
        });

        dlgBox.setNegativeButton("Excluir", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                exibirCaixaConfirmacao(p);
            }
        });

        dlgBox.show();

    }

    public void exibirCaixaConfirmacao(final TalhaoModel p)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(talhaoContext);
        dlgBox.setTitle("Excluindo...");
        dlgBox.setMessage("Tem certeza que deseja excluir esse talhão?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if(cards.size()==1){
                    ExibirMaisUmaCaixa();
                }else{
                    ExcluirTalhao(p.getCod_Talhao(),p.isAplicado());
                }

            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nenhuma ação
            }
        });

        dlgBox.show();

    }


    public void ExibirMaisUmaCaixa(){
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(talhaoContext);
        dlgBox.setTitle("Falha na exclusão");
        dlgBox.setMessage("Sua cultura deve possuir pelo menos um talhão. Caso deseje continuar, exclua a cultura.");
        dlgBox.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    //do nothing
                }
            });
        dlgBox.show();
    }

    public void ExcluirTalhao(Integer CodT, final boolean aplicado){


        String url = "https://mip.software/phpapp/excluirTalhao.php?Cod_Talhao="+ CodT  ;

        RequestQueue queue = Volley.newRequestQueue(talhaoContext);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {
                //Parsing json
                //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                try {
                    JSONObject obj = new JSONObject(response);
                    boolean confirmacao = obj.getBoolean("confirmacao");
                    if(confirmacao){
                        Intent k = new Intent(talhaoContext, Talhoes.class);
                        Toast.makeText(talhaoContext, "Talhão excluído com sucesso!",Toast.LENGTH_LONG).show();
                        k.putExtra("Cod_Propriedade", cod_Propriedade);
                        k.putExtra("nomePropriedade", nomePropriedade);
                        k.putExtra("Cod_Cultura", cod_Cultura);
                        k.putExtra("NomeCultura", nomeCultura);
                        k.putExtra("Aplicado", aplicado);
                        talhaoContext.startActivity(k);
                    }else{
                        Toast.makeText(talhaoContext, "Talhão não excluído! Tente novamente",Toast.LENGTH_LONG).show();
                    }

                } catch (JSONException e) {
                    Toast.makeText(talhaoContext, e.toString(), Toast.LENGTH_LONG).show();
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(talhaoContext,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    public void CalculaDiasPraContagem(int cod_Talhao, final ViewHolder holder, final int position){
        Utils u = new Utils();

        if(!u.isConected(talhaoContext))
        {


            //Toast.makeText(talhaoContext,""+cards.get(1).isAplicado(), Toast.LENGTH_LONG).show();
            //Toast.makeText(talhaoContext,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/diasParaContagem.php?Cod_Talhao=" + cod_Talhao;
            RequestQueue queue = Volley.newRequestQueue(talhaoContext);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            diasPraContagem = obj.getString("DiasPraContagem");
                        }

                        holder.contagem.setText("Dias para a contagem: " + diasPraContagem );
                        if(diasPraContagem.equals("0")){
                            cards.get(position).setAplicado(false);
                        }

                    } catch (JSONException e) {
                        Toast.makeText(talhaoContext, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(talhaoContext,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }
    }

}
